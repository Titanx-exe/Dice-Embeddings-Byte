# Anonymous Code Artifact

This repository contains the code and experiment scripts for the submitted paper.

The repository has been prepared for anonymous review. The main entry point for running experiments is the unified runner in `dicee/run_scripts/`.

---

## Repository layout

```text
.
├── README.md
├── requirements_torch.txt
├── requirements.txt
└── dicee/
    ├── run_scripts/
    │   ├── run_experiment.py
    │   ├── env_single_gpu.sh
    │   └── README.md
    ├── KGs/
    └── Tokenizer/
```

The important files for reviewers are:

```text
dicee/run_scripts/run_experiment.py     # unified script for running experiments
dicee/run_scripts/env_single_gpu.sh     # environment variables for torchrun
dicee/run_scripts/README.md             # detailed experiment instructions
```

The older dataset-specific run scripts are kept for reference, but reviewers should use `run_experiment.py` unless they specifically want to inspect the original scripts.

---

## Setup

From the repository root, create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install the requirements in the following order:

```bash
pip install -r requirements_torch.txt
pip install -r requirements.txt
```

Then move to the experiment directory:

```bash
cd dicee/run_scripts
```

---

## Runtime environment

Before running experiments, source the provided environment file:

```bash
source env_single_gpu.sh
```

This sets the distributed and single-GPU runtime variables used by the experiment commands.

The default setup assumes a single-node, single-GPU run with:

```text
CUDA_VISIBLE_DEVICES=0
MASTER_ADDR=127.0.0.1
MASTER_PORT=29500
```

---

## Quick check

Before launching training, reviewers can list the supported experiments:

```bash
python run_experiment.py --list-experiments
```

A configuration can also be checked without starting training:

```bash
python run_experiment.py \
  --dataset umls \
  --variant normal \
  --setting custom-attention \
  --dry-run
```

The dry run prints the resolved dataset path, tokenizer path, setting, and output directory.

---

## Running an experiment

Experiments are launched from inside `dicee/run_scripts` using `torchrun`.

Example: run the normal UMLS setting with the custom tokenizer and attention layer:

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

The detailed list of supported datasets, variants, and settings is available in:

```text
dicee/run_scripts/README.md
```

---

## Experiment settings

The unified runner supports the following settings:

```text
no-byte
GPT2 tokenizer
custom tokenizer
GPT2 tokenizer + attention
custom tokenizer + attention
```

In the command line, these are selected as:

```text
no-byte
gpt2
custom
gpt2-attention
custom-attention
```

For augmented datasets, the runner selects the corresponding augmented dataset and tokenizer path automatically.

---

## Outputs

Experiment outputs are written under:

```text
dicee/run_scripts/results/
```

Each run creates a structured output folder of the form:

```text
results/<dataset>/<variant>/<setting>/
```

The output folder includes the resolved configuration, the grid-search results, raw reports, and per-run result files.

---

## Notes for reviewers

A full grid can run many model and hyperparameter combinations. For a quick sanity check, start with `--dry-run` and then run one setting before launching additional experiments.

The recommended workflow is:

```bash
cd dicee/run_scripts
source env_single_gpu.sh
python run_experiment.py --list-experiments
python run_experiment.py --dataset umls --variant normal --setting custom-attention --dry-run
torchrun --standalone --nproc_per_node=1 --rdzv_backend=c10d --rdzv_endpoint=127.0.0.1:29500 run_experiment.py --dataset umls --variant normal --setting custom-attention
```

For background execution, additional datasets, and the complete setting matrix, see:

```text
dicee/run_scripts/README.md
```

---

## Anonymous review note

This repository is intended for anonymous peer review. Author-identifying information has been removed from the repository and documentation where possible.
