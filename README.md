# Tokenizing the Graph: Anonymous Code

This repository contains the implementation, datasets, tokenizers, and experiment scripts used for the submitted paper.

For reviewers, the main entry point is the unified experiment runner in:

```text
dicee/run_scripts/run_experiment.py
```

The older dataset-specific scripts are also included in `dicee/run_scripts/` for transparency, but the recommended way to run experiments is through the unified runner.

---

## Repository layout

The repository has several components because it contains both the Dicee codebase and the paper-specific experiment artifacts.

```text
.
├── README.md
├── requirements_torch.txt        # install first
├── requirements.txt              # install second
├── setup.py
├── Dockerfile
├── LICENSE
├── KGs/                          # dataset folder present at the repository root
├── dicee/
│   ├── KGs/                      # datasets used by the experiment scripts
│   ├── Tokenizer/                # custom tokenizer files
│   ├── run_scripts/              # paper experiment scripts
│   │   ├── run_experiment.py     # recommended unified runner
│   │   ├── env_single_gpu.sh     # environment variables for torchrun
│   │   ├── README.md             # detailed run instructions
│   │   └── run_*.py              # original dataset-specific scripts
│   ├── models/                   # knowledge graph embedding models
│   ├── trainer/                  # training backends, including torch DDP
│   ├── scripts/                  # Dicee command-line entry points
│   └── *.py                      # core Dicee implementation
├── docs/                         # package documentation
├── examples/                     # example usage scripts and notebooks
└── tests/                        # package tests
```

A path detail that matters for running the paper experiments: the commands are run from `dicee/run_scripts/`. From there, `../KGs` refers to `dicee/KGs`, and `../Tokenizer` refers to `dicee/Tokenizer`. This matches the paths used by the original run scripts.

---

## What is included

The repository contains:

- the Dicee-based model and training code,
- knowledge graph datasets in `KGs/` and `dicee/KGs/`,
- custom tokenizer files in `dicee/Tokenizer/`,
- original experiment scripts for each dataset,
- a unified runner for reviewer-friendly execution,
- documentation, examples, and tests from the underlying codebase.

The main datasets used by the scripts include:

```text
Countries-S1
Countries-S2
Countries-S3
FB15k-237
NELL-995-h25
NELL-995-h75
NELL-995-h100
UMLS
WN18RR
WN18RR_Text
```

Some experiment variants use augmented datasets. When running an augmented variant, make sure the corresponding augmented dataset folder is present under the dataset directory expected by the runner.

---

## Setup

Create and activate a Python environment from the repository root.

Using `venv`:

```bash
python -m venv .venv
source .venv/bin/activate
```

Or using Conda:

```bash
conda create -n dicee-byte python=3.10 -y
conda activate dicee-byte
```

Install the requirements in this order:

```bash
pip install -r requirements_torch.txt
pip install -r requirements.txt
```

The first file installs the PyTorch build used for these experiments. The second file installs the remaining Python dependencies.

---

## Running the paper experiments

Move into the run-scripts directory:

```bash
cd dicee/run_scripts
```

Load the single-GPU environment variables:

```bash
source env_single_gpu.sh
```

The environment file sets:

```bash
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=lo
export GLOO_SOCKET_IFNAME=lo
export CUDA_VISIBLE_DEVICES=0
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29500
```

List the available experiment options:

```bash
python run_experiment.py --list-experiments
```

Run an experiment with `torchrun`. For example, to run the normal UMLS dataset with the custom tokenizer and attention layer:

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

More detailed examples are in:

```text
dicee/run_scripts/README.md
```

---

## Experiment settings

The unified runner supports five main settings.

| Command-line setting | Meaning |
|---|---|
| `no-byte` | no byte-pair/tokenizer setting |
| `gpt2` | GPT-2/default tokenizer, no attention layer |
| `custom` | dataset-specific custom tokenizer, no attention layer |
| `gpt2-attention` | GPT-2/default tokenizer with attention layer |
| `custom-attention` | dataset-specific custom tokenizer with attention layer |

The default hyperparameter grid is:

```text
models: DistMult, ComplEx, QMult, Keci
embedding dimensions: 32, 64
learning rates: 0.1, 0.01, 0.001
batch size: 512
epochs: 500
```

For non-transformer settings, this gives 24 runs per selected dataset/variant/setting.

---

## Example commands

Normal UMLS with custom tokenizer and attention:

```bash
torchrun --standalone --nproc_per_node=1 --rdzv_backend=c10d --rdzv_endpoint=127.0.0.1:29500 \
  run_experiment.py --dataset umls --variant normal --setting custom-attention
```

Normal FB15k-237 with GPT-2/default tokenizer and attention:

```bash
torchrun --standalone --nproc_per_node=1 --rdzv_backend=c10d --rdzv_endpoint=127.0.0.1:29500 \
  run_experiment.py --dataset fb15k237 --variant normal --setting gpt2-attention
```

Normal NELL h100 with the no-byte baseline:

```bash
torchrun --standalone --nproc_per_node=1 --rdzv_backend=c10d --rdzv_endpoint=127.0.0.1:29500 \
  run_experiment.py --dataset nell-h100 --variant normal --setting no-byte
```

A short smoke test can be run by limiting the grid and number of epochs:

```bash
torchrun --standalone --nproc_per_node=1 --rdzv_backend=c10d --rdzv_endpoint=127.0.0.1:29500 \
  run_experiment.py --dataset umls --variant normal --setting gpt2 --epochs 1 --max-runs 1
```

---

## Running in the background

For long experiments, use `setsid` and redirect output to a log file:

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

Monitor the log with:

```bash
tail -f umls_custom_attention.log
```

---

## Outputs

By default, the unified runner writes outputs under:

```text
dicee/run_scripts/results/
```

Each experiment setting gets its own folder:

```text
results/<dataset>/<variant>/<setting>/
```

The output folder contains:

```text
resolved_config.json       # exact dataset, tokenizer, and run settings
grid_search_results.csv    # one row per model/hyperparameter combination
raw_reports.txt            # full Dicee reports
run*.txt                   # per-run summaries
```

---

## Original scripts

The repository also keeps the original scripts, such as:

```text
run_umls.py
run_umls_augmented.py
run_fb15k_237.py
run_fb15k_237_augmented.py
run_wn18rr.py
run_wn18rr_text_augmented.py
run_nell_h25.py
run_nell_h75.py
run_nell_h100.py
run_nell_h100_augmented.py
run_Countries_S1.py
run_Countries_S2.py
run_Countries_S3.py
run_countries_s1_augmented.py
```

These are useful for inspecting the original experiment structure. For running experiments, reviewers should prefer `run_experiment.py` because it avoids manual edits to source files.

---

## Troubleshooting

### Dataset folder not found

Run commands from inside `dicee/run_scripts/`. The original scripts expect dataset paths relative to this folder.

For example:

```text
../KGs/UMLS
```

from `dicee/run_scripts/` resolves to:

```text
dicee/KGs/UMLS
```

### Tokenizer file not found

Custom-tokenizer settings require the corresponding tokenizer JSON file under `dicee/Tokenizer/`. If a tokenizer is missing, use a GPT-2/default setting such as `gpt2` or `gpt2-attention`, or add the required tokenizer folder.

### NCCL or GLOO error

Re-source the environment file and keep the run to one process/GPU:

```bash
source env_single_gpu.sh
torchrun --standalone --nproc_per_node=1 --rdzv_backend=c10d --rdzv_endpoint=127.0.0.1:29500 ...
```

### Full grid is slow

The default grid is intended for reproducing the paper experiments. For a quick check, use:

```bash
--epochs 1 --max-runs 1
```

---

## Notes for anonymous review

This repository has been prepared as an anonymous code artifact for peer review. The recommended reviewer workflow is:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements_torch.txt
pip install -r requirements.txt
cd dicee/run_scripts
source env_single_gpu.sh
python run_experiment.py --list-experiments
torchrun --standalone --nproc_per_node=1 --rdzv_backend=c10d --rdzv_endpoint=127.0.0.1:29500 run_experiment.py --dataset umls --variant normal --setting custom-attention
```

The detailed experiment-running instructions are in `dicee/run_scripts/README.md`.
