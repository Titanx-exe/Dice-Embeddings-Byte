#!/usr/bin/env python3
"""Unified experiment runner for the Dice Embeddings Byte.

Run from the repository root or from ``dicee/run_scripts``.  The dataset and
Tokenizer paths below are resolved relative to this file, so the command is
less sensitive to the current working directory than the original scripts.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import multiprocessing
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dicee.config import Namespace
from dicee.executer import Execute



DEFAULT_MODELS = ["DistMult", "ComplEx", "QMult", "Keci"]
DEFAULT_EMBEDDING_DIMS = [32, 64]
DEFAULT_LEARNING_RATES = [0.1, 0.01, 0.001]
DEFAULT_BATCH_SIZES = [512]
DEFAULT_NHEAD = [2]
DEFAULT_NUM_LAYERS = [4]


# Dataset/tokenizer paths are relative to dicee/run_scripts.
# ``custom_tokenizer`` is used only for the ``custom`` and
# ``custom-attention`` settings.
EXPERIMENTS: Dict[str, Dict[str, Dict[str, Optional[str]]]] = {
    "umls": {
        "normal": {
            "dataset": "../KGs/UMLS",
            "custom_tokenizer": "../Tokenizer/UMLS_Tokenizer_Path/tokenizer.json",
        },
        "augmented-5": {
            "dataset": "../KGs/UMLS_Augmented_5",
            "custom_tokenizer": "../Tokenizer/UMLS_Augmented_5_Tokenizer_Path/tokenizer.json",
        },
        "augmented-10": {
            "dataset": "../KGs/UMLS_Augmented_10",
            "custom_tokenizer": "../Tokenizer/UMLS_Augmented_10_Tokenizer_Path/tokenizer.json",
        },
    },
    "fb15k237": {
        "normal": {
            "dataset": "../KGs/FB15k-237",
            "custom_tokenizer": "../Tokenizer/FB15k_237_Tokenizer_Path/tokenizer.json",
        },
        "augmented-5": {
            "dataset": "../KGs/FB15k237_Augmented_5",
            "custom_tokenizer": "../Tokenizer/FB15k237_Augmented_5_Tokenizer_Path/tokenizer.json",
        },
        "augmented-10": {
            "dataset": "../KGs/FB15k237_Augmented_10",
            "custom_tokenizer": "../Tokenizer/FB15k237_Augmented_10_Tokenizer_Path/tokenizer.json",
        },
    },
    "wn18rr": {
        "normal": {
            "dataset": "../KGs/WN18RR",
            "custom_tokenizer": "../Tokenizer/WN18RR_Tokenizer_Path/tokenizer.json",
        },
        "augmented-5": {
            "dataset": "../KGs/WN18RR_Text_Augmented_5",
            "custom_tokenizer": "../Tokenizer/WN18RR_Text_Augmented_5_Tokenizer_Path/tokenizer.json",
        },
        "augmented-10": {
            "dataset": "../KGs/WN18RR_Text_Augmented_10",
            "custom_tokenizer": "../Tokenizer/WN18RR_Text_Augmented_10_Tokenizer_Path/tokenizer.json",
        },
    },
    "countries-s1": {
        "normal": {
            "dataset": "../KGs/Countries-S1",
            "custom_tokenizer": "../Tokenizer/Countries_S1_Tokenizer_Path/tokenizer.json",
        },
        "augmented-10": {
            "dataset": "../KGs/Countries_S1_Augmented_10",
            "custom_tokenizer": "../Tokenizer/Countries_S1_Augmented_10_Tokenizer_Path/tokenizer.json",
        },
    },
    "countries-s2": {
        "normal": {
            "dataset": "../KGs/Countries-S2",
            "custom_tokenizer": "../Tokenizer/Countries_S2_Tokenizer_Path/tokenizer.json",
        },
    },
    "countries-s3": {
        "normal": {
            "dataset": "../KGs/Countries-S3",
            "custom_tokenizer": "../Tokenizer/Countries_S3_Tokenizer_Path/tokenizer.json",
        },
    },
    "nell-h25": {
        "normal": {
            "dataset": "../KGs/NELL-995-h25",
            "custom_tokenizer": "../Tokenizer/NELL_995_h25_Tokenizer_Path/tokenizer.json",
        },
    },
    "nell-h75": {
        "normal": {
            "dataset": "../KGs/NELL-995-h75",
            "custom_tokenizer": "../Tokenizer/NELL_995_h75_Tokenizer_Path/tokenizer.json",
        },
    },
    "nell-h100": {
        "normal": {
            "dataset": "../KGs/NELL-995-h100",
            "custom_tokenizer": "../Tokenizer/NELL_995_h100_Tokenizer_Path/tokenizer.json",
        },
        "augmented-5": {
            "dataset": "../KGs/NELL_995_h100_Augmented_5",
            "custom_tokenizer": "../Tokenizer/NELL_995_h100_Augmented_5_Tokenizer_Path/tokenizer.json",
        },
        "augmented-10": {
            "dataset": "../KGs/NELL_995_h100_Augmented_10",
            "custom_tokenizer": "../Tokenizer/NELL_995_h100_Augmented_10_Tokenizer_Path/tokenizer.json",
        },
    },
}


SETTINGS: Dict[str, Dict[str, Any]] = {
    "no-byte": {
        "byte_pair_encoding": False,
        "tokenizer": "none",
        "layer_mode": "none",
    },
    "gpt2": {
        "byte_pair_encoding": True,
        "tokenizer": "none",
        "layer_mode": "none",
    },
    "custom": {
        "byte_pair_encoding": True,
        "tokenizer": "custom",
        "layer_mode": "none",
    },
    "gpt2-attention": {
        "byte_pair_encoding": True,
        "tokenizer": "none",
        "layer_mode": "attention",
    },
    "custom-attention": {
        "byte_pair_encoding": True,
        "tokenizer": "custom",
        "layer_mode": "attention",
    },
}


def resolve_from_script_dir(relative_path: Optional[str]) -> Optional[Path]:
    if not relative_path:
        return None
    return (SCRIPT_DIR / relative_path).resolve()


def comma_list(values: str, cast=str) -> List[Any]:
    return [cast(item.strip()) for item in values.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Dice Embeddings Byte experiments with one parameterized script.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset", choices=sorted(EXPERIMENTS.keys()), help="Dataset family to run.")
    parser.add_argument("--variant", default="normal", help="Dataset variant, e.g. normal, augmented-5, augmented-10.")
    parser.add_argument("--setting", choices=sorted(SETTINGS.keys()), help="Tokenizer/attention setting.")

    parser.add_argument("--models", default=",".join(DEFAULT_MODELS), help="Comma-separated model names.")
    parser.add_argument("--embedding-dims", default="32,64", help="Comma-separated embedding dimensions.")
    parser.add_argument("--learning-rates", default="0.1,0.01,0.001", help="Comma-separated learning rates.")
    parser.add_argument("--batch-sizes", default="512", help="Comma-separated batch sizes.")
    parser.add_argument("--nheads", default="2", help="Comma-separated transformer attention-head counts.")
    parser.add_argument("--num-layers", default="4", help="Comma-separated transformer layer counts.")
    parser.add_argument("--epochs", type=int, default=500, help="Number of training epochs.")

    parser.add_argument(
        "--trainer",
        choices=["torchDDP", "none"],
        default="torchDDP",
        help="Set Dicee trainer. Use torchDDP with the torchrun commands in the README.",
    )
    parser.add_argument(
        "--output-root",
        default="results",
        help="Root directory for grid-search CSVs and per-run result files.",
    )
    parser.add_argument(
        "--max-runs",
        type=int,
        default=None,
        help="Optional cap on the number of hyperparameter combinations to run. Useful for smoke tests.",
    )
    parser.add_argument(
        "--allow-missing-tokenizer",
        action="store_true",
        help="For custom settings, fall back to tokenizer_path=None when the tokenizer file is missing.",
    )
    parser.add_argument(
        "--allow-missing-dataset",
        action="store_true",
        help="Do not fail early when the dataset directory is missing.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print resolved configuration and exit.")
    parser.add_argument("--list-experiments", action="store_true", help="List supported dataset/variant combinations and exit.")
    return parser.parse_args()


def list_experiments() -> None:
    print("Supported dataset variants:\n")
    for dataset_name in sorted(EXPERIMENTS):
        variants = ", ".join(sorted(EXPERIMENTS[dataset_name]))
        print(f"  {dataset_name}: {variants}")
    print("\nSupported settings:")
    for setting_name in sorted(SETTINGS):
        setting = SETTINGS[setting_name]
        print(
            f"  {setting_name}: byte_pair_encoding={setting['byte_pair_encoding']}, "
            f"tokenizer={setting['tokenizer']}, layer_mode={setting['layer_mode']}"
        )


def build_config(args: argparse.Namespace) -> Dict[str, Any]:
    if args.dataset is None or args.setting is None:
        raise SystemExit("--dataset and --setting are required unless --list-experiments is used.")

    dataset_variants = EXPERIMENTS[args.dataset]
    if args.variant not in dataset_variants:
        valid = ", ".join(sorted(dataset_variants))
        raise SystemExit(f"Variant {args.variant!r} is not configured for dataset {args.dataset!r}. Valid variants: {valid}")

    dataset_config = dataset_variants[args.variant]
    setting = SETTINGS[args.setting]

    dataset_dir = resolve_from_script_dir(dataset_config["dataset"])
    if dataset_dir is None:
        raise SystemExit("Internal configuration error: dataset path is missing.")
    if not args.allow_missing_dataset and not dataset_dir.is_dir():
        raise SystemExit(
            f"Dataset directory not found: {dataset_dir}\n"
            "Check that the KGs folder is present, or use --allow-missing-dataset for dry debugging."
        )

    tokenizer_path: Optional[Path] = None
    if setting["tokenizer"] == "custom":
        tokenizer_path = resolve_from_script_dir(dataset_config["custom_tokenizer"])
        if tokenizer_path is None:
            raise SystemExit(f"No custom tokenizer is configured for {args.dataset}/{args.variant}.")
        if not tokenizer_path.is_file():
            if args.allow_missing_tokenizer:
                tokenizer_path = None
            else:
                raise SystemExit(
                    f"Custom tokenizer not found: {tokenizer_path}\n"
                    "Use a GPT2 setting, add the tokenizer file, or pass --allow-missing-tokenizer."
                )

    output_dir = Path(args.output_root) / args.dataset / args.variant / args.setting
    output_dir = output_dir.resolve()

    return {
        "dataset_name": args.dataset,
        "variant": args.variant,
        "setting_name": args.setting,
        "dataset_dir": dataset_dir,
        "byte_pair_encoding": setting["byte_pair_encoding"],
        "tokenizer_path": tokenizer_path,
        "layer_mode": setting["layer_mode"],
        "models": comma_list(args.models, str),
        "embedding_dims": comma_list(args.embedding_dims, int),
        "learning_rates": comma_list(args.learning_rates, float),
        "batch_sizes": comma_list(args.batch_sizes, int),
        "nheads": comma_list(args.nheads, int),
        "num_layers": comma_list(args.num_layers, int),
        "epochs": args.epochs,
        "trainer": None if args.trainer == "none" else args.trainer,
        "output_dir": output_dir,
        "max_runs": args.max_runs,
    }


def make_namespace(
    *,
    model: str,
    dataset_dir: Path,
    embed_dim: int,
    learning_rate: float,
    batch_size: int,
    nhead: Optional[int],
    num_layers: Optional[int],
    layer_mode: str,
    byte_pair_encoding: bool,
    tokenizer_path: Optional[Path],
    epochs: int,
    trainer: Optional[str],
) -> Namespace:
    run_args = Namespace()
    run_args.byte_pair_encoding = byte_pair_encoding
    run_args.tokenizer_path = str(tokenizer_path) if tokenizer_path else None
    run_args.model = model
    run_args.dataset_dir = str(dataset_dir)
    run_args.embedding_dim = embed_dim
    run_args.lr = learning_rate
    run_args.batch_size = batch_size
    run_args.use_attention_layer = layer_mode == "attention"
    run_args.use_transformer_layer = layer_mode == "transformer"

    if trainer:
        run_args.trainer = trainer

    if run_args.use_transformer_layer:
        run_args.transformer_nhead = nhead
        run_args.transformer_num_layers = num_layers
    else:
        run_args.transformer_nhead = None
        run_args.transformer_num_layers = None

    run_args.num_epochs = epochs
    run_args.scoring_technique = "KvsAll"
    run_args.training_technique = "KvsAll"
    return run_args


def run_model(
    *,
    config: Dict[str, Any],
    model: str,
    embed_dim: int,
    learning_rate: float,
    batch_size: int,
    nhead: Optional[int],
    num_layers: Optional[int],
) -> Dict[str, Any]:
    layer_mode = config["layer_mode"]
    print(
        "Running: "
        f"model={model}, dataset={config['dataset_dir']}, lr={learning_rate}, "
        f"dim={embed_dim}, bs={batch_size}, nhead={nhead}, layers={num_layers}, "
        f"mode={layer_mode}, setting={config['setting_name']}"
    )

    run_args = make_namespace(
        model=model,
        dataset_dir=config["dataset_dir"],
        embed_dim=embed_dim,
        learning_rate=learning_rate,
        batch_size=batch_size,
        nhead=nhead,
        num_layers=num_layers,
        layer_mode=layer_mode,
        byte_pair_encoding=config["byte_pair_encoding"],
        tokenizer_path=config["tokenizer_path"],
        epochs=config["epochs"],
        trainer=config["trainer"],
    )

    reports = Execute(run_args).start()
    train_mrr = reports["Train"]["MRR"]
    test_mrr = reports["Test"]["MRR"]
    print("Train MRR:", train_mrr)
    print("Test  MRR:", test_mrr)
    return reports


def combination_iter(config: Dict[str, Any]) -> Iterable[Tuple[str, int, float, int, Optional[int], Optional[int]]]:
    layer_mode = config["layer_mode"]
    if layer_mode == "transformer":
        for model, lr, dim, bs, nhead, num_layers in itertools.product(
            config["models"],
            config["learning_rates"],
            config["embedding_dims"],
            config["batch_sizes"],
            config["nheads"],
            config["num_layers"],
        ):
            yield model, dim, lr, bs, nhead, num_layers
    else:
        for model, lr, dim, bs in itertools.product(
            config["models"],
            config["learning_rates"],
            config["embedding_dims"],
            config["batch_sizes"],
        ):
            yield model, dim, lr, bs, None, None


def write_single_result(
    output_dir: Path,
    run_index: int,
    config: Dict[str, Any],
    model: str,
    embed_dim: int,
    learning_rate: float,
    batch_size: int,
    nhead: Optional[int],
    num_layers: Optional[int],
    train_mrr: Any,
    test_mrr: Any,
    reports: Dict[str, Any],
) -> Path:
    filename = (
        f"run{run_index:03d}_"
        f"model-{model}_lr-{learning_rate}_dim-{embed_dim}_bs-{batch_size}_"
        f"mode-{config['layer_mode']}.txt"
    )
    path = output_dir / filename.replace("/", "-")
    with path.open("w", encoding="utf-8") as f_run:
        f_run.write(f"Dataset: {config['dataset_name']}\n")
        f_run.write(f"Variant: {config['variant']}\n")
        f_run.write(f"Dataset path: {config['dataset_dir']}\n")
        f_run.write(f"Setting: {config['setting_name']}\n")
        f_run.write(f"Byte pair encoding: {config['byte_pair_encoding']}\n")
        f_run.write(f"Tokenizer path: {config['tokenizer_path']}\n")
        f_run.write(f"Model: {model}\n")
        f_run.write(f"Learning Rate: {learning_rate}\n")
        f_run.write(f"Embed Dim: {embed_dim}\n")
        f_run.write(f"Batch Size: {batch_size}\n")
        f_run.write(f"nhead: {nhead if nhead is not None else 'NA'}\n")
        f_run.write(f"num_layers: {num_layers if num_layers is not None else 'NA'}\n")
        f_run.write(f"layer_mode: {config['layer_mode']}\n")
        f_run.write(f"trainer: {config['trainer']}\n\n")
        f_run.write(f"Train MRR: {train_mrr}\n")
        f_run.write(f"Test  MRR: {test_mrr}\n\n")
        f_run.write("Full report:\n")
        f_run.write(json.dumps(reports, indent=2, default=str))
        f_run.write("\n")
    return path


def main() -> int:
    multiprocessing.freeze_support()
    args = parse_args()

    if args.list_experiments:
        list_experiments()
        return 0

    config = build_config(args)

    printable_config = {
        **config,
        "dataset_dir": str(config["dataset_dir"]),
        "tokenizer_path": str(config["tokenizer_path"]) if config["tokenizer_path"] else None,
        "output_dir": str(config["output_dir"]),
    }

    if args.dry_run:
        print(json.dumps(printable_config, indent=2, default=str))
        return 0

    output_dir: Path = config["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    config_path = output_dir / "resolved_config.json"
    config_path.write_text(json.dumps(printable_config, indent=2), encoding="utf-8")

    grid_path = output_dir / "grid_search_results.csv"
    raw_report_path = output_dir / "raw_reports.txt"

    fieldnames = [
        "run_index",
        "model",
        "dataset",
        "variant",
        "setting",
        "dataset_path",
        "tokenizer_path",
        "byte_pair_encoding",
        "lr",
        "embed_dim",
        "batch_size",
        "nhead",
        "num_layers",
        "layer_mode",
        "trainer",
        "train_mrr",
        "test_mrr",
    ]

    run_count = 0
    with grid_path.open("w", newline="", encoding="utf-8") as grid_file, raw_report_path.open(
        "w", encoding="utf-8"
    ) as raw_report_file:
        writer = csv.DictWriter(grid_file, fieldnames=fieldnames)
        writer.writeheader()

        for run_index, (model, dim, lr, bs, nhead, num_layers) in enumerate(combination_iter(config), start=1):
            if config["max_runs"] is not None and run_count >= config["max_runs"]:
                break

            reports = run_model(
                config=config,
                model=model,
                embed_dim=dim,
                learning_rate=lr,
                batch_size=bs,
                nhead=nhead,
                num_layers=num_layers,
            )
            run_count += 1

            train_mrr = reports["Train"]["MRR"]
            test_mrr = reports["Test"]["MRR"]

            writer.writerow(
                {
                    "run_index": run_index,
                    "model": model,
                    "dataset": config["dataset_name"],
                    "variant": config["variant"],
                    "setting": config["setting_name"],
                    "dataset_path": str(config["dataset_dir"]),
                    "tokenizer_path": str(config["tokenizer_path"]) if config["tokenizer_path"] else "",
                    "byte_pair_encoding": config["byte_pair_encoding"],
                    "lr": lr,
                    "embed_dim": dim,
                    "batch_size": bs,
                    "nhead": nhead if nhead is not None else "NA",
                    "num_layers": num_layers if num_layers is not None else "NA",
                    "layer_mode": config["layer_mode"],
                    "trainer": config["trainer"] or "",
                    "train_mrr": train_mrr,
                    "test_mrr": test_mrr,
                }
            )
            grid_file.flush()

            write_single_result(
                output_dir=output_dir,
                run_index=run_index,
                config=config,
                model=model,
                embed_dim=dim,
                learning_rate=lr,
                batch_size=bs,
                nhead=nhead,
                num_layers=num_layers,
                train_mrr=train_mrr,
                test_mrr=test_mrr,
                reports=reports,
            )

            raw_report_file.write(f"Run {run_index}\n")
            raw_report_file.write(json.dumps(reports, indent=2, default=str))
            raw_report_file.write("\n\n")
            raw_report_file.flush()

    print(f"Finished {run_count} run(s).")
    print(f"Grid results: {grid_path}")
    print(f"Raw reports:  {raw_report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
