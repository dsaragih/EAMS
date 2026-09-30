"""Training script using PyTorch Lightning and Hydra for configuration.

This refactored version replaces the custom training loop with PyTorch Lightning's
Trainer API and uses Hydra for configuration management.

Usage:
    # Train with default config
    python train_lightning.py

    # Override specific parameters
    python train_lightning.py model.lr=0.001 data.batch_size=32

    # Use different config
    python train_lightning.py --config-name=experiment_config
"""

import os

import hydra
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    LearningRateMonitor,
    EarlyStopping,
)
from pytorch_lightning.loggers import Logger
import torch

# Import our modules
from test_lightning import run_test
from util.config import register_custom_omegaconf_resolvers
from util.experiment_test_utils import build_robustness_variants, find_best_checkpoint


def setup_callbacks(cfg: DictConfig, fold: int | None = None):
    """Setup PyTorch Lightning callbacks."""
    callbacks = []

    # Model checkpointing - save best models
    checkpoint_callback_avg = ModelCheckpoint(
        dirpath=os.path.join(
            cfg.paths.checkpoints_dir, f"fold_{fold}" if fold is not None else ""
        ),
        filename="ckpt_{epoch:03d}",
        monitor="val/avg/iou",
        mode="max",
        save_last=False,
    )
    callbacks.append(checkpoint_callback_avg)

    # Learning rate monitor
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    callbacks.append(lr_monitor)

    # Optional: Early stopping
    # early_stop = EarlyStopping(
    #     monitor='val/avg/dice',
    #     patience=5,
    #     mode='max',
    #     verbose=True
    # )
    # callbacks.append(early_stop)

    return callbacks


def setup_logger(cfg: DictConfig, fold: int | None = None) -> list[Logger]:
    """Setup TensorBoard logger."""
    loggers = []
    if not cfg.get("logger", None):
        return loggers
    for lg_type, lg_conf in cfg.logger.items():
        loggers.append(
            hydra.utils.instantiate(
                lg_conf,
                name="" if fold is None else f"{cfg.paths.experiment_name}_fold_{fold}",
            )
        )
    return loggers


def _clone_cfg(cfg: DictConfig) -> DictConfig:
    return OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))


def _resolve_fold_checkpoint_path(
    cfg: DictConfig,
    fold: int,
    checkpoint_callback: ModelCheckpoint | None,
) -> str:
    if checkpoint_callback is not None and checkpoint_callback.best_model_path:
        return checkpoint_callback.best_model_path

    checkpoint_dir = os.path.join(cfg.paths.checkpoints_dir, f"fold_{fold}")
    return find_best_checkpoint(checkpoint_dir, metric="avg")


def build_trainer_kwargs(
    cfg: DictConfig,
    *,
    model,
    callbacks,
    logger,
) -> dict:
    trainer_kwargs = OmegaConf.to_container(cfg.trainer, resolve=True)
    assert isinstance(trainer_kwargs, dict)

    if not bool(getattr(model, "automatic_optimization", True)):
        trainer_clip_val = trainer_kwargs.get("gradient_clip_val", 0.0)
        if hasattr(model, "configure_manual_gradient_clipping"):
            model.configure_manual_gradient_clipping(trainer_clip_val)
        del trainer_kwargs["gradient_clip_val"]

    trainer_kwargs["callbacks"] = callbacks
    trainer_kwargs["logger"] = logger
    return trainer_kwargs


def run_post_training_robustness_suite(
    cfg: DictConfig,
    *,
    checkpoint_path: str,
    fold: int,
) -> None:
    if not cfg.robustness_eval.get("enabled", False):
        return

    base_results_dir = cfg.paths.results_dir
    fold_results_dir = os.path.join(
        base_results_dir,
        cfg.robustness_eval.get("results_subdir", "robustness"),
        f"fold_{fold}",
    )
    variants = build_robustness_variants(cfg.robustness_eval, cfg.transforms)
    fail_on_error = cfg.robustness_eval.get("fail_on_error", False)

    print("\n" + "=" * 80)
    print(f"POST-TRAIN ROBUSTNESS EVALUATION FOR FOLD {fold + 1}")
    print("=" * 80)

    for variant in variants:
        condition_name = variant["condition_name"]
        print(f"\nRunning robustness condition: {condition_name}")

        test_cfg = _clone_cfg(cfg)
        test_cfg.checkpoint_path = checkpoint_path
        test_cfg.paths.results_dir = fold_results_dir
        test_cfg.transforms.condition_name = condition_name
        base_test_transforms = OmegaConf.to_container(
            test_cfg.transforms.get("test_transforms", []), resolve=False
        )
        variant_test_transforms = variant.get("test_transforms", [])
        test_cfg.transforms.test_transforms = OmegaConf.create(
            list(variant_test_transforms) + list(base_test_transforms)
        )

        if hasattr(test_cfg.data, "current_fold"):
            test_cfg.data.current_fold = fold

        try:
            run_test(test_cfg)
        except Exception as exc:
            print(
                f"Robustness condition '{condition_name}' failed with {type(exc).__name__}: {exc}"
            )
            if fail_on_error:
                raise


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    """Main training function."""

    # Handle debug mode
    debug_mode = cfg.get("debug", False)
    if debug_mode:
        print("\n" + "!" * 80)
        print("DEBUG MODE ENABLED - Fast iteration settings activated:")
        print("  - Checkpointing: DISABLED")
        print("  - Logging: DISABLED")
        print("  - Data workers: 0")
        print("  - Deterministic: True")
        print("  - Limit batches: 2 per epoch")
        print("!" * 80 + "\n")

        # Override settings for debug mode
        cfg.trainer.enable_checkpointing = False
        cfg.trainer.devices = 1
        cfg.trainer.limit_train_batches = 2
        cfg.trainer.limit_val_batches = 2
        # cfg.trainer.deterministic = True
        cfg.trainer.benchmark = False
        cfg.trainer.max_epochs = 2
        cfg.data.num_workers = 0

    # Validate k-fold configuration
    k_folds = cfg.get("k_folds", 1)

    print(f"\n{'='*80}")
    print(f"K-FOLD CROSS-VALIDATION: {k_folds} folds")
    print(f"{'='*80}\n")

    # Set seed for reproducibility
    if cfg.seed is not None:
        pl.seed_everything(cfg.seed, workers=True)

    # Set matmul precision if specified
    if "matmul_precision" in cfg and cfg.matmul_precision is not None:
        torch.set_float32_matmul_precision(cfg.matmul_precision)
        print(f"Set float32 matmul precision to: {cfg.matmul_precision}")

    # K-fold cross-validation loop
    fold_results = []
    OmegaConf.resolve(cfg)  # Resolve any interpolations
    # Print configuration
    print("=" * 80)
    print("Configuration:")
    print(OmegaConf.to_yaml(cfg))
    print("=" * 80)

    for fold in range(k_folds):
        print("\n" + "=" * 80)
        print(f"FOLD {fold + 1}/{k_folds}")
        print("=" * 80)

        # Initialize DataModule with current fold
        print(f"\nInitializing DataModule for fold {fold + 1}...")

        datamodule = hydra.utils.instantiate(
            cfg.data,
            seed=cfg.seed,
            current_fold=fold,
        )

        # Setup data to get nclasses and input_nc
        datamodule.setup("fit")

        # Initialize Model
        print(f"\nInitializing Model for fold {fold + 1}...")
        # Resolve any interpolations in the config before passing to model
        model = hydra.utils.instantiate(cfg.model, _recursive_=False)

        # Setup callbacks and logger (skip in debug mode)
        debug_mode = cfg.get("debug", False)
        if debug_mode:
            callbacks = []
            logger = False
            print("\nDebug mode: Skipping callbacks and logger setup")
        else:
            # Create fold-specific logger and callbacks
            # Finish previous logger if it exists (e.g., wandb.finish())
            if fold > 0 and logger and len(logger) > 0:
                for lg in logger:
                    if hasattr(lg, "experiment") and hasattr(lg.experiment, "finish"):
                        lg.experiment.finish()

            callbacks = setup_callbacks(cfg, fold)
            logger = setup_logger(cfg, fold)

        # Initialize Trainer
        print(f"\nInitializing Trainer for fold {fold + 1}...")
        trainer = pl.Trainer(
            **build_trainer_kwargs(
                cfg,
                model=model,
                callbacks=callbacks,
                logger=logger,
            )
        )

        # Print model summary
        print("\nModel Summary:")
        print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(
            f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}"
        )

        # Train the model
        print(f"\nStarting training for fold {fold + 1}...")
        trainer.fit(model, datamodule=datamodule)

        # Store fold results
        if not debug_mode:
            checkpoint_callback = [
                cb for cb in callbacks if isinstance(cb, ModelCheckpoint)
            ][0]
            checkpoint_path = _resolve_fold_checkpoint_path(
                cfg, fold=fold, checkpoint_callback=checkpoint_callback
            )
            fold_result = {
                "fold": fold,
                "best_checkpoint": checkpoint_path,
                "best_score": (
                    checkpoint_callback.best_model_score.item()
                    if checkpoint_callback.best_model_score is not None
                    else None
                ),
            }
            fold_results.append(fold_result)

            print(f"\nFold {fold + 1} completed:")
            print(f"  Best checkpoint: {fold_result['best_checkpoint']}")
            print(
                f"  Best score: {fold_result['best_score']:.4f}"
                if fold_result["best_score"]
                else "  No score recorded"
            )

            run_post_training_robustness_suite(
                cfg,
                checkpoint_path=checkpoint_path,
                fold=fold,
            )

    # Print summary of all folds
    print("\n" + "=" * 80)
    print("K-FOLD CROSS-VALIDATION COMPLETED!")
    print("=" * 80)

    if not debug_mode and len(fold_results) > 0:
        print("\nResults per fold:")
        valid_scores = [
            r["best_score"] for r in fold_results if r["best_score"] is not None
        ]
        for result in fold_results:
            score_str = (
                f"{result['best_score']:.4f}"
                if result["best_score"] is not None
                else "N/A"
            )
            print(f"  Fold {result['fold'] + 1}: {score_str}")
            print(f"    Checkpoint: {result['best_checkpoint']}")

        if len(valid_scores) > 0:
            print(
                f"\nAverage validation score across folds: {sum(valid_scores) / len(valid_scores):.4f}"
            )
            print(f"Std deviation: {torch.tensor(valid_scores).std().item():.4f}")

    print("=" * 80)


if __name__ == "__main__":
    register_custom_omegaconf_resolvers()
    main()
