"""Testing script using PyTorch Lightning and Hydra for configuration.

This refactored version uses PyTorch Lightning's test interface.

Usage:
    # Test with default config and best checkpoint
    python test_lightning.py

    # Test with specific checkpoint
    python test_lightning.py checkpoint_path=/path/to/checkpoint.ckpt

    # Override other parameters
    python test_lightning.py data.batch_size=1 paths.results_dir=./custom_results

    # Scratch
"""

import os
from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
import torch

# Import our modules
from util.config import register_custom_omegaconf_resolvers
from util.experiment_test_utils import (
    clone_cfg,
    find_best_checkpoint,
    infer_fold_from_path,
    load_test_cfg_from_experiment_dir,
)


def _sync_test_runtime_config(cfg: DictConfig) -> DictConfig:
    # logger.info(f"Cfg before synchronization:\n{OmegaConf.to_yaml(cfg)}")
    if hasattr(cfg, "data") and hasattr(cfg, "transforms"):
        cfg.data.transforms = clone_cfg(cfg.transforms)
    return cfg


def prepare_test_cfg(
    cfg: DictConfig,
) -> tuple[DictConfig, dict[str, Path] | None, list[str]]:
    experiment_dir = cfg.get("experiment_dir", None)
    if not experiment_dir:
        return _sync_test_runtime_config(cfg), None, []

    runtime_task_overrides = []
    hydra_cfg = HydraConfig.get()
    if hydra_cfg is not None and hydra_cfg.overrides is not None:
        runtime_task_overrides = list(hydra_cfg.overrides.task)

    prepared_cfg, artifacts, saved_overrides = load_test_cfg_from_experiment_dir(
        experiment_dir,
        runtime_cfg=cfg,
        runtime_task_overrides=runtime_task_overrides,
    )
    return _sync_test_runtime_config(prepared_cfg), artifacts, saved_overrides


def run_test(cfg: DictConfig):
    """Run the Lightning test flow for a fully prepared config."""

    cfg = _sync_test_runtime_config(cfg)

    condition_name = cfg.transforms.get("condition_name", None)
    results_dir = cfg.paths.results_dir
    if condition_name:
        results_dir = os.path.join(results_dir, str(condition_name))

    debug_mode = cfg.get("debug", False)
    if debug_mode:
        print("\n" + "!" * 80)
        print("DEBUG MODE ENABLED - Fast testing settings activated:")
        print("  - Data workers: 0")
        print("  - Limit batches: 2")
        print("!" * 80 + "\n")

        cfg.trainer.devices = 1
        cfg.trainer.limit_test_batches = 2
        cfg.data.num_workers = 0

    print("=" * 80)
    print("Test Configuration:")
    print(OmegaConf.to_yaml(cfg))
    print("=" * 80)

    if cfg.seed is not None:
        pl.seed_everything(cfg.seed, workers=True)

    if "matmul_precision" in cfg and cfg.matmul_precision is not None:
        torch.set_float32_matmul_precision(cfg.matmul_precision)
        print(f"Set float32 matmul precision to: {cfg.matmul_precision}")

    checkpoint_path = cfg.get("checkpoint_path", None)
    if checkpoint_path is None:
        checkpoint_dir = os.path.join(cfg.paths.checkpoints_dir)
        if os.path.exists(checkpoint_dir):
            try:
                checkpoint_path = find_best_checkpoint(checkpoint_dir, metric="avg")
                print(f"\nUsing best checkpoint: {checkpoint_path}")
            except FileNotFoundError as e:
                print(f"\nError: {e}")
                print("Please specify checkpoint_path in config or command line.")
                return []
        else:
            print(f"\nError: Checkpoint directory not found: {checkpoint_dir}")
            print("Please specify checkpoint_path in config or command line.")
            return []
    else:
        if not os.path.exists(checkpoint_path):
            print(f"\nError: Checkpoint not found: {checkpoint_path}")
            return []
        print(f"\nUsing specified checkpoint: {checkpoint_path}")

    inferred_fold = infer_fold_from_path(checkpoint_path)
    datamodule_kwargs = {"seed": cfg.seed}
    if inferred_fold is not None:
        datamodule_kwargs["current_fold"] = inferred_fold

    print("\nInitializing DataModule...")
    datamodule = hydra.utils.instantiate(cfg.data, **datamodule_kwargs)
    print("\nLoading model from checkpoint...")
    model_class = hydra.utils.get_class(cfg.model._target_)
    try:
        model = model_class.load_from_checkpoint(
            checkpoint_path, results_dir=results_dir, weights_only=False
        )
    except Exception:
        model = model_class.load_from_checkpoint(
            checkpoint_path,
            results_dir=results_dir,
            network_config=cfg.model.network_config,
            metrics_dict=cfg.model.metrics_dict,
            weights_only=False,
        )

    model.results_dir = results_dir
    model.save_pt_results = bool(
        cfg.model.get("save_pt_results", getattr(model, "save_pt_results", True))
    )

    os.makedirs(results_dir, exist_ok=True)
    print(f"Results will be saved to: {results_dir}")

    print("\nInitializing Trainer for testing...")
    trainer = pl.Trainer(
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        precision=cfg.trainer.precision,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=True,
    )

    print("\nRunning test...")
    test_results = trainer.test(model, datamodule=datamodule)
    print(f"\nDetailed results saved to: {results_dir}")
    return test_results


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    """Main testing function."""
    prepared_cfg, artifacts, saved_overrides = prepare_test_cfg(cfg)
    if artifacts is not None:
        print(f"Recovered experiment config from: {artifacts['config_path']}")
        if saved_overrides:
            print("Recovered training overrides:")
            for override in saved_overrides:
                print(f"  - {override}")

    run_test(prepared_cfg)


if __name__ == "__main__":
    register_custom_omegaconf_resolvers()
    main()
