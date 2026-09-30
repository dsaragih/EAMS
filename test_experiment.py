"""Batch test checkpoints from a saved experiment directory.

Usage:
    python test_experiment.py /path/to/experiment
    python test_experiment.py /path/to/experiment --fold 0
"""

import argparse
import os
from pathlib import Path
from numbers import Number
from collections import defaultdict
from statistics import pstdev

from omegaconf import DictConfig, OmegaConf

from test_lightning import run_test
from util.config import register_custom_omegaconf_resolvers
from util.experiment_test_utils import (
    build_robustness_variants,
    clone_cfg,
    discover_checkpoint_paths,
    load_test_cfg_from_experiment_dir,
    load_robustness_eval_config,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run test_lightning.py logic across every checkpoint in an experiment "
            "directory. Expected layout: experiment_dir/checkpoints/fold_i/*.ckpt"
        )
    )
    parser.add_argument(
        "experiment_dir",
        help="Saved experiment directory containing outputs/, checkpoints/, logs/, and results/.",
    )
    parser.add_argument(
        "--fold",
        type=int,
        default=None,
        help="Optional 0-based fold index to restrict evaluation to one fold.",
    )
    parser.add_argument(
        "--results-subdir",
        default="checkpoint_eval",
        help="Subdirectory under the experiment results root used for per-checkpoint test outputs.",
    )
    parser.add_argument(
        "--robustness-eval-config",
        default=None,
        help=(
            "Optional robustness_eval config name or YAML path. "
            "Example: rz15_rz40_refx"
        ),
    )
    save_group = parser.add_mutually_exclusive_group()
    save_group.add_argument(
        "--save-pt-results",
        dest="save_pt_results",
        action="store_true",
        help="Save per-sample heavy .pt result artifacts during evaluation.",
    )
    save_group.add_argument(
        "--no-save-pt-results",
        dest="save_pt_results",
        action="store_false",
        help="Skip per-sample heavy .pt result artifacts during evaluation.",
    )
    parser.set_defaults(save_pt_results=False)
    return parser.parse_args()


def _build_checkpoint_results_dir(
    base_results_dir: str,
    checkpoint_path: Path,
    fold: int,
    results_subdir: str,
) -> str:
    return os.path.join(
        base_results_dir,
        results_subdir,
        f"fold_{fold}",
        checkpoint_path.stem,
    )


def _prepare_checkpoint_cfg(
    base_cfg: DictConfig,
    checkpoint_path: Path,
    fold: int,
    base_results_dir: str,
    results_subdir: str,
    experiment_dir: str,
    save_pt_results: bool,
) -> DictConfig:
    checkpoint_cfg = clone_cfg(base_cfg)
    checkpoint_cfg.experiment_dir = experiment_dir
    checkpoint_cfg.checkpoint_path = str(checkpoint_path)
    checkpoint_cfg.paths.results_dir = _build_checkpoint_results_dir(
        base_results_dir,
        checkpoint_path,
        fold,
        results_subdir,
    )
    checkpoint_cfg.model.save_pt_results = bool(save_pt_results)

    if hasattr(checkpoint_cfg, "data") and "current_fold" in checkpoint_cfg.data:
        checkpoint_cfg.data.current_fold = fold

    return checkpoint_cfg


def _run_checkpoint_robustness_suite(
    checkpoint_cfg: DictConfig,
    fold: int,
    robustness_eval_cfg: DictConfig,
) -> dict[str, list[dict[str, float]]]:
    variants = build_robustness_variants(
        robustness_eval_cfg,
        checkpoint_cfg.transforms,
    )
    base_results_dir = checkpoint_cfg.paths.results_dir
    results_subdir = robustness_eval_cfg.get("results_subdir", "robustness")
    aggregated_runs: dict[str, list[dict[str, float]]] = defaultdict(list)

    print("\n" + "=" * 80)
    print(f"ROBUSTNESS EVALUATION FOR FOLD {fold + 1}")
    print("=" * 80)

    for variant in variants:
        condition_name = variant["condition_name"]
        print(f"\nRunning robustness condition: {condition_name}")

        variant_cfg = clone_cfg(checkpoint_cfg)
        variant_cfg.robustness_eval = clone_cfg(robustness_eval_cfg)
        variant_cfg.paths.results_dir = os.path.join(base_results_dir, results_subdir)
        variant_cfg.transforms.condition_name = condition_name

        base_test_transforms = OmegaConf.to_container(
            variant_cfg.transforms.get("test_transforms", []), resolve=False
        )
        variant_test_transforms = variant.get("test_transforms", [])
        variant_cfg.transforms.test_transforms = OmegaConf.create(
            list(variant_test_transforms) + list(base_test_transforms)
        )

        if hasattr(variant_cfg.data, "current_fold"):
            variant_cfg.data.current_fold = fold

        aggregated = run_test(variant_cfg)
        if isinstance(aggregated, list):
            aggregated_runs[condition_name].extend(
                item for item in aggregated if isinstance(item, dict)
            )
        elif isinstance(aggregated, dict):
            aggregated_runs[condition_name].append(aggregated)

    return dict(aggregated_runs)


def _print_metric_means(
    summary_name: str, aggregated_runs: list[dict[str, float]]
) -> None:
    print(f"\n{summary_name}")
    metric_keys = sorted(
        {
            key
            for row in aggregated_runs
            for key, value in row.items()
            if isinstance(value, Number)
        }
    )
    for key in metric_keys:
        values = [
            row[key] for row in aggregated_runs if isinstance(row.get(key), Number)
        ]
        if values:
            mean_value = sum(values) / len(values)
            std_value = pstdev(values)
            print(f"{key}: {mean_value} ({std_value})")


def main() -> None:
    args = parse_args()
    base_cfg, artifacts, saved_overrides = load_test_cfg_from_experiment_dir(
        args.experiment_dir,
    )
    robustness_eval_cfg = None
    if args.robustness_eval_config:
        robustness_eval_cfg = load_robustness_eval_config(args.robustness_eval_config)

    checkpoint_runs = discover_checkpoint_paths(
        artifacts["checkpoints_dir"],
        fold=args.fold,
    )

    print(f"Recovered experiment config from: {artifacts['config_path']}")
    if saved_overrides:
        print("Recovered training overrides:")
        for override in saved_overrides:
            print(f"  - {override}")
    if robustness_eval_cfg is not None:
        print(f"Using robustness eval config: {args.robustness_eval_config}")

    print(f"Discovered {len(checkpoint_runs)} checkpoint(s) to evaluate.")
    aggregated_runs: list[dict[str, float]] = []
    robustness_aggregated_runs: dict[str, list[dict[str, float]]] = defaultdict(list)

    for fold, checkpoint_path in checkpoint_runs:
        print("\n" + "=" * 80)
        print(f"Evaluating fold {fold}: {checkpoint_path}")
        print("=" * 80)
        checkpoint_cfg = _prepare_checkpoint_cfg(
            base_cfg,
            checkpoint_path=checkpoint_path,
            fold=fold,
            base_results_dir=str(artifacts["results_dir"]),
            results_subdir=args.results_subdir,
            experiment_dir=args.experiment_dir,
            save_pt_results=args.save_pt_results,
        )
        if robustness_eval_cfg is not None and robustness_eval_cfg.get(
            "enabled", False
        ):
            fold_robustness_runs = _run_checkpoint_robustness_suite(
                checkpoint_cfg,
                fold=fold,
                robustness_eval_cfg=robustness_eval_cfg,
            )
            for condition_name, runs in fold_robustness_runs.items():
                robustness_aggregated_runs[condition_name].extend(runs)
        else:
            aggregated = run_test(checkpoint_cfg)
            if isinstance(aggregated, list):
                aggregated_runs.extend(
                    item for item in aggregated if isinstance(item, dict)
                )
            elif isinstance(aggregated, dict):
                aggregated_runs.append(aggregated)

    if aggregated_runs:
        _print_metric_means("Mean metrics across evaluated folds:", aggregated_runs)

    for condition_name in sorted(robustness_aggregated_runs):
        _print_metric_means(
            f"Mean metrics across evaluated folds for robustness condition: {condition_name}",
            robustness_aggregated_runs[condition_name],
        )


if __name__ == "__main__":
    register_custom_omegaconf_resolvers()
    main()
