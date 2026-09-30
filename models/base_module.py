"""PyTorch Lightning Module for MeshGraphCNN."""

import hydra
import json
import math
from contextlib import nullcontext
from typing import Optional, Any

import pytorch_lightning as pl
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch_geometric.data import Batch
from util.losses import BaseLoss

import numpy as np
import os

from loguru import logger as log

from data import const
from data.transforms.transforms import _apply_rigid_transform

from .frame_averaging import (
    FrameAveragingSettings,
    FrameAveragingBatch,
    average_frame_tensor,
    build_frame_averaging_batch,
    collapse_non_equivariant_encoder_output,
    frame_averaging_metadata,
    infer_frame_tensor_level,
    prepare_frame_energy_batch,
    sample_frame_indices,
    select_frame_tensor,
    uniform_frame_weights,
    weighted_frame_tensor,
)
from .learned_canonicalization import CanonicalizationResult, canonicalize_batch
from .direct_canonicalization import (
    DirectCanonicalizationResult,
    direct_canonicalize_batch,
)
from .rigid_batch_utils import (
    propagate_dynamic_node_attrs,
    reuse_rigid_transform_invariant_features,
)
from .post_processing.base import (
    BasePostProcessor,
    NoOpPostProcessor,
    PostProcessInput,
)


class BaseClassifierLightning(pl.LightningModule):
    """Lightning Module wrapping the MeshGraphCNN model.

    This module handles:
    - Model initialization
    - Optimization configuration
    - Metrics logging
    """

    def __init__(
        self,
        # Network configuration (Hydra config dict or actual network instance)
        network: Optional[Any] = None,
        network_config: Optional[DictConfig] = None,
        encoder: Optional[Any] = None,
        encoder_config: Optional[DictConfig] = None,
        decoder: Optional[Any] = None,
        decoder_config: Optional[DictConfig] = None,
        energy_encoder: Optional[Any] = None,
        energy_encoder_config: Optional[DictConfig] = None,
        energy_decoder: Optional[Any] = None,
        energy_decoder_config: Optional[DictConfig] = None,
        canonicalizer_encoder: Optional[Any] = None,
        canonicalizer_encoder_config: Optional[DictConfig] = None,
        canonicalizer_decoder: Optional[Any] = None,
        canonicalizer_decoder_config: Optional[DictConfig] = None,
        # Optimizer
        lr: float = 0.01,
        beta1: float = 0.9,
        weight_decay: float = 0.0,
        optimizer_name: str = "adamw",
        # Learning rate scheduler
        lr_policy: str = "lambda",
        lr_decay_iters: int = 50,
        lr_step_gamma: float = 0.1,
        niter: int = 100,
        niter_decay: int = 500,
        plateau_factor: float = 0.6,
        plateau_threshold: float = 0.01,
        plateau_patience: int = 2,
        min_lr: float = 1e-5,
        # Loss
        batched_input: bool = True,
        edge_to_pc: str = "midpoint",
        loss_dict: dict[str, dict] = {},
        loss_coefficients: dict[str, dict] = {},
        metrics_dict: list[dict] = None,
        # Classes / metrics behavior
        classes: str = None,
        include_background_metrics: bool = False,
        # Paths
        train_lossweight_dir: str = None,
        results_dir: str = "./results",
        save_pt_results: bool = True,
        post_process: Optional[Any] = None,
        output_embedding_key: str = "node_embedding",
        equivariance_regularization: Optional[Any] = None,
        frame_averaging: Optional[Any] = None,
        learned_canonicalization: Optional[Any] = None,
        direct_canonicalization: Optional[Any] = None,
        # Profiling
        profile_training: bool = False,
        profiling: Optional[Any] = None,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters(
            ignore=[
                "network",
                "encoder",
                "decoder",
                "energy_encoder",
                "energy_decoder",
                "canonicalizer_encoder",
                "canonicalizer_decoder",
            ]
        )

        # Store hyperparameters
        self.lr = lr
        self.beta1 = beta1
        self.weight_decay = weight_decay
        self.optimizer_name = optimizer_name
        self.lr_policy = lr_policy
        self.lr_decay_iters = lr_decay_iters
        self.lr_step_gamma = lr_step_gamma
        self.niter = niter
        self.niter_decay = niter_decay
        self.plateau_factor = plateau_factor
        self.plateau_threshold = plateau_threshold
        self.plateau_patience = plateau_patience
        self.min_lr = min_lr
        self.edge_to_pc = edge_to_pc
        self.profile_training = profile_training
        self.equivariance_regularization = equivariance_regularization or {}
        self.frame_averaging = frame_averaging or {}
        self.frame_averaging_settings = FrameAveragingSettings(self.frame_averaging)
        self.learned_canonicalization = learned_canonicalization or {}
        self.direct_canonicalization = direct_canonicalization or {}
        self.profiling = profiling or {}
        self._manual_main_grad_clip_val = 0.0

        # Profiling stats
        if self.profile_training:
            from util.profiler import GPUTimingStats

            self.timing_stats = GPUTimingStats()
        else:
            self.timing_stats = None

        # Loss weights for multi-scale supervision
        self.losses = loss_dict
        self.loss_coefficients = loss_coefficients
        self.losses = self.configure_losses(batched_input=batched_input)
        log.info(f"Configured losses: {self.losses}")
        # Paths
        self.train_lossweight_dir = train_lossweight_dir
        self.results_dir = results_dir
        self.save_pt_results = bool(save_pt_results)
        os.makedirs(results_dir, exist_ok=True)
        self.post_process = self._configure_post_process(post_process)

        # Instantiate network using Hydra
        if network is not None:
            # Network instance provided directly
            self.net = network
        elif (
            encoder is not None
            or encoder_config is not None
            or decoder is not None
            or decoder_config is not None
        ):
            if network_config is not None:
                log.warning(
                    "Provide either network_config or encoder/decoder configs, not both"
                )
                log.warning(
                    "Defaulting to network_config and ignoring encoder/decoder configs"
                )
                self.net = instantiate(network_config)
            else:
                self.net = self._build_composed_network(
                    encoder=encoder,
                    encoder_config=encoder_config,
                    decoder=decoder,
                    decoder_config=decoder_config,
                    output_embedding_key=output_embedding_key,
                )
        elif network_config is not None:
            # Instantiate from Hydra config
            self.net = instantiate(network_config)
        else:
            raise ValueError(
                "Either 'network' (network instance) or 'network_config' (Hydra config) must be provided"
            )

        self.energy_net = None
        if any(
            self._component_is_configured(value)
            for value in (
                energy_encoder,
                energy_encoder_config,
                energy_decoder,
                energy_decoder_config,
            )
        ):
            self.energy_net = self._build_energy_network(
                encoder=energy_encoder,
                encoder_config=energy_encoder_config,
                decoder=energy_decoder,
                decoder_config=energy_decoder_config,
                output_embedding_key=output_embedding_key,
            )
        self.canonicalizer_net = None
        if any(
            self._component_is_configured(value)
            for value in (
                canonicalizer_encoder,
                canonicalizer_encoder_config,
                canonicalizer_decoder,
                canonicalizer_decoder_config,
            )
        ):
            self.canonicalizer_net = self._build_canonicalizer_network(
                encoder=canonicalizer_encoder,
                encoder_config=canonicalizer_encoder_config,
                decoder=canonicalizer_decoder,
                decoder_config=canonicalizer_decoder_config,
                output_embedding_key=output_embedding_key,
            )
        if self._manual_energy_optimization_enabled():
            self.automatic_optimization = False

        # Metrics tracking
        self.validation_step_outputs = []
        self.test_step_outputs = []

        # Configure metrics
        self.metric_config = metrics_dict

        self.classes = []
        if classes is not None:
            # Expect it to be a path to a txt file listing class names, one per line
            if os.path.isfile(classes):
                with open(classes, "r") as f:
                    self.classes = [line.strip() for line in f if line.strip()]
            else:
                raise ValueError(f"Provided 'classes' path is not a file: {classes}")

        if not isinstance(self.classes, list) or len(self.classes) == 0:
            raise ValueError("'classes' must be a non-empty list of class names")

        self.include_background_metrics = include_background_metrics
        self.metric_class_names = (
            self.classes if self.include_background_metrics else self.classes[1:]
        )
        self.configure_metrics()

    def iter_metric_classes(self):
        start_class_id = 0 if self.include_background_metrics else 1
        for class_id, class_name in enumerate(
            self.metric_class_names, start=start_class_id
        ):
            yield class_id, class_name

    def _configure_post_process(
        self, post_process: Optional[Any]
    ) -> BasePostProcessor | None:
        if post_process is None:
            return None
        if isinstance(post_process, BasePostProcessor):
            return post_process
        if isinstance(post_process, (DictConfig, dict)):
            configured = instantiate(post_process)
        else:
            configured = post_process

        if configured is None:
            return None
        if not isinstance(configured, BasePostProcessor):
            raise TypeError("post_process must resolve to a BasePostProcessor instance")
        if isinstance(configured, NoOpPostProcessor) and not configured.enabled:
            return configured
        log.info(f"Configured post-process: {configured.__class__.__name__}")
        return configured

    @property
    def encoder(self):
        return getattr(self.net, "encoder", None)

    @property
    def decoder(self):
        return getattr(self.net, "decoder", None)

    @property
    def energy_encoder(self):
        return getattr(self.energy_net, "encoder", None)

    @property
    def energy_decoder(self):
        return getattr(self.energy_net, "decoder", None)

    @property
    def canonicalizer_encoder(self):
        return getattr(self.canonicalizer_net, "encoder", None)

    @property
    def canonicalizer_decoder(self):
        return getattr(self.canonicalizer_net, "decoder", None)

    @staticmethod
    def _component_is_configured(value: Any) -> bool:
        if value is None:
            return False
        if isinstance(value, (DictConfig, dict)):
            return len(value) > 0
        return True

    def _build_energy_network(
        self,
        encoder: Optional[Any],
        encoder_config: Optional[DictConfig],
        decoder: Optional[Any],
        decoder_config: Optional[DictConfig],
        output_embedding_key: str = "node_embedding",
    ):
        if encoder is not None:
            energy_encoder = encoder
        else:
            energy_encoder_kwargs = {}
            if self._frame_averaging_energy_input_source() == "embedding":
                main_encoder_output_dim = getattr(self.encoder, "output_dim", None)
                if main_encoder_output_dim is None:
                    raise ValueError(
                        "frame_averaging.energy_input_source='embedding' requires "
                        "the main encoder to expose output_dim."
                    )
                energy_encoder_kwargs["input_dim"] = int(main_encoder_output_dim)
            energy_encoder = instantiate(encoder_config, **energy_encoder_kwargs)

        if decoder is None:
            decoder_kwargs = {}
            if hasattr(energy_encoder, "output_dim"):
                decoder_kwargs["node_dim"] = energy_encoder.output_dim
                decoder_kwargs["input_dim"] = energy_encoder.output_dim
            if hasattr(energy_encoder, "face_output_dim"):
                decoder_kwargs["face_dim"] = energy_encoder.face_output_dim
            if hasattr(energy_encoder, "edges_input_dim"):
                decoder_kwargs["edge_attr_dim"] = energy_encoder.edges_input_dim
            energy_decoder = instantiate(decoder_config, **decoder_kwargs)
        else:
            energy_decoder = decoder

        from .networks.encoder_decoder import EncoderDecoderNetwork

        return EncoderDecoderNetwork(
            encoder=energy_encoder,
            decoder=energy_decoder,
            output_embedding_key=output_embedding_key,
        )

    def _build_canonicalizer_network(
        self,
        encoder: Optional[Any],
        encoder_config: Optional[DictConfig],
        decoder: Optional[Any],
        decoder_config: Optional[DictConfig],
        output_embedding_key: str = "node_embedding",
    ):
        if encoder is None:
            canonicalizer_encoder = instantiate(encoder_config)
        else:
            canonicalizer_encoder = encoder

        if decoder is None:
            decoder_kwargs = {}
            if hasattr(canonicalizer_encoder, "output_dim"):
                decoder_kwargs["node_dim"] = canonicalizer_encoder.output_dim
                decoder_kwargs["input_dim"] = canonicalizer_encoder.output_dim
            canonicalizer_decoder = instantiate(decoder_config, **decoder_kwargs)
        else:
            canonicalizer_decoder = decoder

        from .networks.encoder_decoder import EncoderDecoderNetwork

        return EncoderDecoderNetwork(
            encoder=canonicalizer_encoder,
            decoder=canonicalizer_decoder,
            output_embedding_key=output_embedding_key,
        )

    def _build_composed_network(
        self,
        encoder: Optional[Any],
        encoder_config: Optional[DictConfig],
        decoder: Optional[Any],
        decoder_config: Optional[DictConfig],
        output_embedding_key: str = "node_embedding",
    ):
        if encoder is None and encoder_config is None:
            raise ValueError("encoder or encoder_config must be provided")
        if decoder is None and decoder_config is None:
            raise ValueError("decoder or decoder_config must be provided")

        if encoder is None:
            encoder = instantiate(encoder_config)

        if decoder is None:
            decoder_kwargs = {}
            if hasattr(encoder, "output_dim"):
                decoder_kwargs["node_dim"] = encoder.output_dim
                decoder_kwargs["input_dim"] = encoder.output_dim
            if hasattr(encoder, "face_output_dim"):
                decoder_kwargs["face_dim"] = encoder.face_output_dim
            if hasattr(encoder, "edges_input_dim"):
                decoder_kwargs["edge_attr_dim"] = encoder.edges_input_dim
            decoder = instantiate(decoder_config, **decoder_kwargs)

        from .networks.encoder_decoder import EncoderDecoderNetwork

        return EncoderDecoderNetwork(
            encoder=encoder,
            decoder=decoder,
            output_embedding_key=output_embedding_key,
        )

    def configure_losses(self, batched_input: bool = True):
        """Configure loss functions if needed.
        dict of the form {'output_name': loss_str}
        """
        log.info(f"Configuring losses with batched_input={batched_input}")
        losses = {}

        for output_key, loss_conf in self.losses.items():
            # Allow optional loss entries (e.g., embedding loss disabled)
            if output_key.startswith("_") or loss_conf is None:
                continue

            if isinstance(loss_conf, dict) and len(loss_conf) == 0:
                continue
            if isinstance(loss_conf, DictConfig) and len(loss_conf.keys()) == 0:
                continue

            if isinstance(loss_conf, BaseLoss):
                losses[output_key] = loss_conf
            elif isinstance(loss_conf, DictConfig) or isinstance(loss_conf, dict):
                losses[output_key] = hydra.utils.instantiate(
                    loss_conf, batched_input=batched_input
                )
                # Check if instantiated loss is a BaseLoss or a dictionary
                if isinstance(losses[output_key], dict) or isinstance(
                    losses[output_key], DictConfig
                ):
                    # Take first value if dict
                    if len(losses[output_key]) == 0:
                        continue
                    losses[output_key] = list(losses[output_key].values())[0]

                assert isinstance(
                    losses[output_key], BaseLoss
                ), f"Instantiated loss for '{output_key}' is not a BaseLoss instance: {losses[output_key]}"
            else:
                raise ValueError(
                    f"Invalid loss configuration for '{output_key}': {loss_conf}"
                )
        return losses

    def configure_metrics(self):
        """Configure metrics for evaluation.

        Sets up metrics for each class and stage (train/val/test).
        Metrics are categorized as:
        - Label metrics: operate on predicted vs true labels (e.g., Dice coefficient)
        - Point cloud metrics: operate on point clouds (e.g., Chamfer distance)

        Metrics are stored as attributes: {stage}_{class_name}_{metric_name}
        e.g., 'val_ligament_dice', 'test_ridge_chamfer'
        """

        self.metric_names = []

        assert self.metric_config is not None, "metric_config must be provided"

        for metric_name, metric_conf in self.metric_config.items():
            # Determine if this is a label metric or point cloud metric
            is_label_metric = metric_name in const.LABEL_METRICS
            is_pc_metric = metric_name in const.POINT_CLOUD_METRICS

            if not (is_label_metric or is_pc_metric):
                log.warning(
                    f"Unknown metric type for {metric_name}, assuming label metric"
                )
                is_label_metric = True

            # Create metrics for each class and stage
            for class_name in self.metric_class_names:
                for stage in {"train", "val", "test"}:
                    # Instantiate metric if config provided, otherwise will use built-in method
                    if metric_conf is not None:
                        metric = hydra.utils.instantiate(metric_conf)
                    else:
                        metric = None  # Will use built-in methods

                    attr_name = f"{stage}_{class_name}_{metric_name}"
                    setattr(self, attr_name, metric)

            if metric_name not in self.metric_names:
                self.metric_names.append(metric_name)

    def update_metrics(
        self,
        log_dict: dict,
        class_name: str,
        stage: str,
        pred_class: torch.Tensor,
        labels: torch.Tensor,
        pred_coords: torch.Tensor,
        gt_coords: torch.Tensor,
    ):
        """Update metrics dictionary for a single sample and class.

        This function computes metrics for one sample/mesh and one class,
        and updates the provided log_dict. Subclasses can call this repeatedly
        with their own batching logic.

        Args:
            log_dict: Dictionary to update with computed metrics
            pred_class: Predicted class labels for this sample
            labels: True class labels for this sample
            class_name: Name of the class (e.g., 'ligament')
            class_id: ID of the class (e.g., 1)
            stage: One of 'train', 'val', 'test'
            edges: (2, E) edge indices for point cloud metrics (optional)
            coords: (N, 3) vertex coordinates for point cloud metrics (optional)

        Returns:
            Dictionary with computed metric values for this sample/class
        """

        metrics = {}

        # Compute label-based metrics
        for metric_name in self.metric_names:
            attr_name = f"{stage}_{class_name}_{metric_name}"
            if not hasattr(self, attr_name):
                continue

            metric_obj = getattr(self, attr_name)

            # If no metric object was configured, skip.
            if metric_obj is None:
                continue

            if metric_name in const.LABEL_METRICS:
                # Use dice_coefficient function or metric
                value = metric_obj(pred_class, labels)
                value = value.item() if torch.is_tensor(value) else value
                metrics[metric_name] = value

            elif metric_name in const.POINT_CLOUD_METRICS:
                value = metric_obj(pred_coords, gt_coords)
                if value is None:
                    continue
                value = value.item() if torch.is_tensor(value) else value
                metrics[metric_name] = value

        # Update log_dict with computed metrics
        for metric_name, value in metrics.items():
            key = f"{stage}/{class_name}/{metric_name}"
            assert key not in log_dict
            log_dict[key] = value

        return metrics

    def _create_binary_mask(
        self, labels: torch.Tensor, target_class: int
    ) -> torch.Tensor:
        """Create binary mask for a specific class.

        Args:
            labels: Tensor of class labels
            target_class: Target class ID to create mask for

        Returns:
            Binary tensor (1 for target class, 0 otherwise)
        """
        mask = labels.clone()
        # Set all non-target classes to 0, target class to 1
        mask = (mask == target_class).long()
        return mask

    def compute_loss(
        self,
        output: dict[str, torch.Tensor],
        labels: torch.Tensor | dict[torch.Tensor],
        w_loss=None,
        data=None,
    ):
        """Compute configured losses from model outputs and optional supervision."""
        losses = {}
        for output_key in self.losses:
            loss_obj = self.losses[output_key]
            kwargs = {}
            if loss_obj.is_weighted_loss:
                kwargs["w_loss"] = w_loss

            for attr in loss_obj.required_attributes:
                if data is not None and hasattr(data, attr):
                    kwargs[attr] = getattr(data, attr)
                else:
                    raise ValueError(
                        f"Required attribute '{attr}' for loss '{output_key}' not found in {data}."
                    )

            # Helpful optional attributes
            if data is not None and hasattr(data, "batch") and "batch" not in kwargs:
                kwargs["batch"] = getattr(data, "batch")

            supervision = self._resolve_loss_supervision(
                output_key=output_key,
                loss_obj=loss_obj,
                supervision=labels,
            )

            losses[output_key] = loss_obj(output[output_key], supervision, **kwargs)

        if self.loss_coefficients:
            for output_key, coeff in self.loss_coefficients.items():
                losses[output_key] = losses[output_key] * coeff

        return losses

    def _resolve_loss_supervision(
        self,
        *,
        output_key: str,
        loss_obj: BaseLoss,
        supervision: torch.Tensor | dict[str, torch.Tensor] | None,
    ) -> torch.Tensor | None:
        if not getattr(loss_obj, "requires_supervision", True):
            return None

        if supervision is None:
            raise ValueError(
                f"Loss '{output_key}' requires supervision but none was provided."
            )

        if isinstance(supervision, torch.Tensor):
            return supervision

        if output_key in supervision:
            return supervision[output_key]

        supervision_key = getattr(loss_obj, "supervision_key", None)
        if supervision_key is not None and supervision_key in supervision:
            return supervision[supervision_key]

        if "shared" in supervision:
            return supervision["shared"]

        if "pred" in supervision:
            return supervision["pred"]

        raise ValueError(
            f"Could not resolve supervision for loss '{output_key}'. Available keys: {list(supervision.keys())}"
        )

    def _cfg_get(self, cfg: Any, key: str, default: Any = None) -> Any:
        if cfg is None:
            return default
        if isinstance(cfg, DictConfig):
            return cfg.get(key, default)
        if isinstance(cfg, dict):
            return cfg.get(key, default)
        return getattr(cfg, key, default)

    def _timed_section(self, name: str):
        if self.timing_stats is None:
            return nullcontext()
        return self.timing_stats.time(name)

    def _profiling_cfg_get(self, key: str, default: Any) -> Any:
        return self._cfg_get(self.profiling, key, default)

    def _profiling_enabled_for_stage(self, stage: str) -> bool:
        if not self.profile_training or self.timing_stats is None:
            return False
        if stage == "val":
            return bool(self._profiling_cfg_get("include_validation", False))
        return stage == "train"

    def _profiling_report_every_n_steps(self) -> int:
        return max(1, int(self._profiling_cfg_get("report_every_n_steps", 10)))

    def _profiling_warmup_steps(self) -> int:
        return max(0, int(self._profiling_cfg_get("warmup_steps", 0)))

    def _profiling_reset_after_report(self) -> bool:
        return bool(self._profiling_cfg_get("reset_after_report", True))

    def _profiling_dump_json(self) -> bool:
        return bool(self._profiling_cfg_get("dump_json", True))

    def _dump_timing_summary(self, stage: str, batch_idx: int) -> None:
        if self.timing_stats is None:
            return
        payload = {
            "stage": stage,
            "epoch": int(self.current_epoch),
            "batch_idx": int(batch_idx),
            "global_step": int(self.global_step),
            "summary": self.timing_stats.get_summary(),
        }
        output_path = os.path.join(
            self.results_dir,
            f"timing_{stage}_epoch{int(self.current_epoch):03d}_step{int(self.global_step):06d}.json",
        )
        with open(output_path, "w") as f:
            json.dump(payload, f, indent=2)

    def _maybe_report_timing(self, stage: str, batch_idx: int, logger_fn) -> None:
        if not self._profiling_enabled_for_stage(stage):
            return

        step_idx = int(batch_idx) + 1
        warmup_steps = self._profiling_warmup_steps()
        if warmup_steps > 0 and step_idx == warmup_steps:
            self.timing_stats.reset()
            return
        if step_idx <= warmup_steps:
            return
        if step_idx % self._profiling_report_every_n_steps() != 0:
            return

        logger_fn("\n" + self.timing_stats.get_percentage_report())
        if self._profiling_dump_json():
            self._dump_timing_summary(stage=stage, batch_idx=batch_idx)
        if self._profiling_reset_after_report():
            self.timing_stats.reset()

    def equivariance_regularization_enabled(self) -> bool:
        return bool(self._cfg_get(self.equivariance_regularization, "enabled", False))

    def _equivariance_transform_config(self) -> Any:
        return self._cfg_get(self.equivariance_regularization, "transform", {})

    def _equivariance_rotation_axis(
        self, axis: Any, *, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        if isinstance(axis, str):
            if axis == "random":
                axis_tensor = torch.randn(3, device=device, dtype=dtype)
            elif axis == "x":
                axis_tensor = torch.tensor([1.0, 0.0, 0.0], device=device, dtype=dtype)
            elif axis == "y":
                axis_tensor = torch.tensor([0.0, 1.0, 0.0], device=device, dtype=dtype)
            elif axis == "z":
                axis_tensor = torch.tensor([0.0, 0.0, 1.0], device=device, dtype=dtype)
            else:
                raise ValueError(
                    f"Unsupported equivariance rotation axis '{axis}'. "
                    "Expected one of x, y, z, random, or a length-3 vector."
                )
        else:
            axis_tensor = torch.as_tensor(axis, device=device, dtype=dtype)
            if axis_tensor.shape != (3,):
                raise ValueError(
                    "Equivariance rotation axis must be a length-3 vector, "
                    f"got shape={tuple(axis_tensor.shape)}."
                )

        norm = axis_tensor.norm().clamp_min(1e-12)
        return axis_tensor / norm

    def _sample_equivariance_degrees(
        self, degrees: Any, *, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        if isinstance(degrees, (int, float)):
            low = -float(degrees)
            high = float(degrees)
        else:
            if len(degrees) != 2:
                raise ValueError(
                    "Equivariance rotation degrees must be a scalar or a two-value range."
                )
            low = float(degrees[0])
            high = float(degrees[1])
        if low > high:
            raise ValueError(
                f"Equivariance rotation degree range must be ordered, got {degrees!r}."
            )
        return torch.empty((), device=device, dtype=dtype).uniform_(low, high)

    def _sample_equivariance_rotation_matrix(
        self, batch, transform_cfg: Any | None = None
    ) -> torch.Tensor:
        if transform_cfg is None:
            transform_cfg = self._equivariance_transform_config()
        transform_type = self._cfg_get(transform_cfg, "type", "random_rotation")
        if transform_type != "random_rotation":
            raise ValueError(
                "Only equivariance_regularization.transform.type=random_rotation "
                f"is supported, got {transform_type!r}."
            )

        pos = batch.pos
        axis = self._cfg_get(transform_cfg, "axis", "random")
        degrees = self._cfg_get(transform_cfg, "degrees", (-180.0, 180.0))
        axis_tensor = self._equivariance_rotation_axis(
            axis, device=pos.device, dtype=pos.dtype
        )
        angle = self._sample_equivariance_degrees(
            degrees, device=pos.device, dtype=pos.dtype
        )
        theta = angle * (math.pi / 180.0)
        kx, ky, kz = axis_tensor.unbind()
        zero = kx.new_zeros(())
        skew = torch.stack(
            [
                torch.stack([zero, -kz, ky]),
                torch.stack([kz, zero, -kx]),
                torch.stack([-ky, kx, zero]),
            ]
        )
        identity = torch.eye(3, device=pos.device, dtype=pos.dtype)
        return (
            identity
            + torch.sin(theta) * skew
            + (1.0 - torch.cos(theta)) * (skew @ skew)
        )

    def build_equivariance_batch(self, batch):
        transformed_samples = []
        samples = batch.to_data_list()
        self._propagate_dynamic_node_attrs(batch, samples)
        for sample in samples:
            transformed = sample.clone()
            rotation = self._sample_equivariance_rotation_matrix(transformed)
            transformed = _apply_rigid_transform(transformed, matrix=rotation)
            transformed_samples.append(transformed)

        transformed_batch = Batch.from_data_list(transformed_samples)
        if hasattr(self, "featuriser"):
            reused = self._reuse_rigid_transform_invariant_features(
                source_batch=batch,
                target_batch=transformed_batch,
                samples=samples,
                repeats=1,
            )
            if not reused:
                transformed_batch = self.featuriser(transformed_batch)
        return transformed_batch

    def learned_canonicalization_enabled(self) -> bool:
        enabled = bool(self._cfg_get(self.learned_canonicalization, "enabled", False))
        if not enabled:
            return False
        if bool(self._cfg_get(self.frame_averaging, "enabled", False)):
            raise ValueError(
                "learned_canonicalization and frame_averaging are mutually exclusive."
            )
        if self.equivariance_regularization_enabled():
            raise ValueError(
                "learned_canonicalization and equivariance_regularization are mutually exclusive."
            )
        if self.energy_net is None:
            raise ValueError(
                "learned_canonicalization requires energy_encoder_config and "
                "energy_decoder_config."
            )
        if self._frame_averaging_energy_input_source() != "framed_batch":
            raise ValueError(
                "learned_canonicalization requires energy_input_source='framed_batch'."
            )
        return True

    def _learned_canonicalization_cfg(self, key: str, default: Any = None) -> Any:
        return self._cfg_get(self.learned_canonicalization, key, default)

    def _canonicalization_result(self, batch) -> CanonicalizationResult:
        return canonicalize_batch(
            batch=batch,
            energy_net=self.energy_net,
            featuriser=getattr(self, "featuriser", None),
            propagate_dynamic_node_attrs=self._propagate_dynamic_node_attrs,
            reuse_features=self._reuse_rigid_transform_invariant_features,
            steps=int(self._learned_canonicalization_cfg("steps", 3)),
            step_size=float(self._learned_canonicalization_cfg("step_size", 0.1)),
            max_step_angle=float(
                self._learned_canonicalization_cfg("max_step_angle", 0.25)
            ),
            differentiable=bool(self.training),
        )

    @staticmethod
    def _canonicalization_metadata(
        result: CanonicalizationResult,
    ) -> dict[str, torch.Tensor]:
        return {
            "canonicalization_pos": result.batch.pos,
            "canonicalization_normal": result.batch.normal,
            "canonicalization_rotation": result.rotations,
            "canonicalization_seed_energies": result.seed_energies,
            "canonicalization_energy_history": result.energy_history,
            "canonicalization_selected_seed": result.selected_seed,
            "canonicalization_eigengaps": result.eigengaps,
            "canonicalization_energy_gap": result.energy_gap,
        }

    def _learned_canonicalization_rotation_config(self) -> Any:
        return self._learned_canonicalization_cfg(
            "consistency_transform",
            {"type": "random_rotation", "degrees": [-180.0, 180.0], "axis": "random"},
        )

    def _build_learned_canonicalization_pair(self, batch) -> tuple[Batch, torch.Tensor]:
        samples = batch.to_data_list()
        self._propagate_dynamic_node_attrs(batch, samples)
        transformed_samples = []
        rotations = []
        transform_cfg = self._learned_canonicalization_rotation_config()
        for sample in samples:
            rotation = self._sample_equivariance_rotation_matrix(sample, transform_cfg)
            rotations.append(rotation)
            transformed_samples.append(
                _apply_rigid_transform(sample.clone(), matrix=rotation)
            )
        transformed_batch = Batch.from_data_list(transformed_samples)
        if not self._reuse_rigid_transform_invariant_features(
            source_batch=batch,
            target_batch=transformed_batch,
            samples=samples,
            repeats=1,
        ):
            transformed_batch = self.featuriser(transformed_batch)
        return transformed_batch, torch.stack(rotations)

    def learned_canonicalization_consistency(
        self, *, original_output: dict[str, torch.Tensor], batch
    ) -> dict[str, torch.Tensor]:
        transformed_batch, input_rotations = self._build_learned_canonicalization_pair(batch)
        transformed = self._canonicalization_result(transformed_batch)
        original_pos = original_output["canonicalization_pos"]
        original_normal = original_output["canonicalization_normal"]
        pos_mse = torch.mean((transformed.batch.pos - original_pos) ** 2)
        normal_mse = torch.mean((transformed.batch.normal - original_normal) ** 2)
        pose_error = torch.linalg.matrix_norm(
            transformed.rotations - input_rotations @ original_output["canonicalization_rotation"],
            ord="fro",
            dim=(-2, -1),
        ).mean()
        return {
            "loss": 0.5 * (pos_mse + normal_mse),
            "position_rms": pos_mse.sqrt(),
            "normal_rms": normal_mse.sqrt(),
            "pose_error": pose_error,
            "weight": original_pos.new_tensor(
                float(self._learned_canonicalization_cfg("consistency_weight", 0.05))
            ),
        }

    def direct_canonicalization_enabled(self) -> bool:
        enabled = bool(self._cfg_get(self.direct_canonicalization, "enabled", False))
        if not enabled:
            return False
        if bool(self._cfg_get(self.frame_averaging, "enabled", False)):
            raise ValueError(
                "direct_canonicalization and frame_averaging are mutually exclusive."
            )
        if bool(self._cfg_get(self.learned_canonicalization, "enabled", False)):
            raise ValueError(
                "direct_canonicalization and learned_canonicalization are mutually exclusive."
            )
        if self.equivariance_regularization_enabled():
            raise ValueError(
                "direct_canonicalization and equivariance_regularization are mutually exclusive."
            )
        if self.energy_net is not None:
            raise ValueError(
                "direct_canonicalization cannot use energy_encoder_config or "
                "energy_decoder_config."
            )
        if self.canonicalizer_net is None:
            raise ValueError(
                "direct_canonicalization requires canonicalizer_encoder_config and "
                "canonicalizer_decoder_config."
            )
        return True

    def _direct_canonicalization_cfg(self, key: str, default: Any = None) -> Any:
        return self._cfg_get(self.direct_canonicalization, key, default)

    def _direct_canonicalization_result(self, batch) -> DirectCanonicalizationResult:
        return direct_canonicalize_batch(
            batch=batch,
            canonicalizer_net=self.canonicalizer_net,
            featuriser=getattr(self, "featuriser", None),
            propagate_dynamic_node_attrs=self._propagate_dynamic_node_attrs,
            reuse_features=self._reuse_rigid_transform_invariant_features,
        )

    @staticmethod
    def _direct_canonicalization_metadata(
        result: DirectCanonicalizationResult,
    ) -> dict[str, torch.Tensor]:
        return {
            "direct_canonicalization_pos": result.batch.pos,
            "direct_canonicalization_normal": result.batch.normal,
            "direct_canonicalization_rotation": result.rotations,
            "direct_canonicalization_raw_matrix": result.raw_matrices,
            "direct_canonicalization_singular_values": result.singular_values,
            "direct_canonicalization_projection_residual": result.projection_residuals,
        }

    def _direct_canonicalization_consistency_cfg(
        self, key: str, default: Any = None
    ) -> Any:
        consistency = self._direct_canonicalization_cfg("consistency", {})
        return self._cfg_get(consistency, key, default)

    def direct_canonicalization_consistency_enabled(self) -> bool:
        if not self.direct_canonicalization_enabled():
            return False
        enabled = bool(self._direct_canonicalization_consistency_cfg("enabled", False))
        weight = float(self._direct_canonicalization_consistency_cfg("weight", 0.0))
        if weight < 0.0:
            raise ValueError("direct_canonicalization.consistency.weight must be non-negative.")
        return enabled and weight > 0.0

    def _build_direct_canonicalization_pair(self, batch) -> tuple[Batch, torch.Tensor]:
        samples = batch.to_data_list()
        self._propagate_dynamic_node_attrs(batch, samples)
        transformed_samples = []
        rotations = []
        transform_cfg = self._direct_canonicalization_consistency_cfg(
            "transform",
            {"type": "random_rotation", "degrees": [-180.0, 180.0], "axis": "random"},
        )
        for sample in samples:
            rotation = self._sample_equivariance_rotation_matrix(sample, transform_cfg)
            rotations.append(rotation)
            transformed_samples.append(
                _apply_rigid_transform(sample.clone(), matrix=rotation)
            )
        transformed_batch = Batch.from_data_list(transformed_samples)
        if not self._reuse_rigid_transform_invariant_features(
            source_batch=batch,
            target_batch=transformed_batch,
            samples=samples,
            repeats=1,
        ):
            transformed_batch = self.featuriser(transformed_batch)
        return transformed_batch, torch.stack(rotations)

    def direct_canonicalization_consistency(
        self, *, original_output: dict[str, torch.Tensor], batch
    ) -> dict[str, torch.Tensor]:
        transformed_batch, input_rotations = self._build_direct_canonicalization_pair(batch)
        transformed = self._direct_canonicalization_result(transformed_batch)
        original_pos = original_output["direct_canonicalization_pos"]
        original_normal = original_output["direct_canonicalization_normal"]
        pos_mse = torch.mean((transformed.batch.pos - original_pos) ** 2)
        normal_mse = torch.mean((transformed.batch.normal - original_normal) ** 2)
        pose_error = torch.linalg.matrix_norm(
            transformed.rotations
            - input_rotations @ original_output["direct_canonicalization_rotation"],
            ord="fro",
            dim=(-2, -1),
        ).mean()
        return {
            "loss": 0.5 * (pos_mse + normal_mse),
            "position_rms": pos_mse.sqrt(),
            "normal_rms": normal_mse.sqrt(),
            "pose_error": pose_error,
            "weight": original_pos.new_tensor(
                float(self._direct_canonicalization_consistency_cfg("weight", 0.0))
            ),
        }

    def _direct_canonicalization_id_bias_cfg(
        self, key: str, default: Any = None
    ) -> Any:
        id_bias = self._direct_canonicalization_cfg("id_bias", {})
        return self._cfg_get(id_bias, key, default)

    def direct_canonicalization_id_bias_enabled(self) -> bool:
        if not self.direct_canonicalization_enabled():
            return False
        enabled = bool(self._direct_canonicalization_id_bias_cfg("enabled", False))
        weight = float(self._direct_canonicalization_id_bias_cfg("weight", 0.0))
        if weight < 0.0:
            raise ValueError("direct_canonicalization.id_bias.weight must be non-negative.")
        return enabled and weight > 0.0

    def direct_canonicalization_id_bias(
        self, original_output: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Return the native-pose identity prior for predicted rotations."""
        rotations = original_output["direct_canonicalization_rotation"]
        identity = torch.eye(3, device=rotations.device, dtype=rotations.dtype)
        mse = torch.mean((rotations - identity) ** 2)
        return {
            "loss": mse,
            "weight": rotations.new_tensor(
                float(self._direct_canonicalization_id_bias_cfg("weight", 0.0))
            ),
        }

    def compute_equivariance_consistency_loss(
        self,
        original_output: dict[str, torch.Tensor],
        transformed_output: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        compare_key = self._cfg_get(
            self.equivariance_regularization, "compare_key", "pred"
        )
        if compare_key not in original_output:
            raise ValueError(
                f"Original output is missing equivariance compare_key={compare_key!r}."
            )
        if compare_key not in transformed_output:
            raise ValueError(
                f"Transformed output is missing equivariance compare_key={compare_key!r}."
            )

        original = original_output[compare_key]
        transformed = transformed_output[compare_key]
        if original.shape != transformed.shape:
            raise ValueError(
                "Equivariance consistency outputs must have matching shapes: "
                f"original={tuple(original.shape)}, transformed={tuple(transformed.shape)}."
            )
        return torch.mean((transformed - original) ** 2)

    def equivariance_consistency_weight(self) -> float:
        return float(
            self._cfg_get(self.equivariance_regularization, "consistency_weight", 0.0)
        )

    def equivariance_transformed_supervision_weight(self) -> float:
        return float(
            self._cfg_get(
                self.equivariance_regularization,
                "transformed_supervision_weight",
                1.0,
            )
        )

    def frame_averaging_enabled(self, stage: str | None = None) -> bool:
        del stage
        enabled = bool(self._cfg_get(self.frame_averaging, "enabled", False))
        if enabled and bool(self._cfg_get(self.learned_canonicalization, "enabled", False)):
            raise ValueError(
                "frame_averaging and learned_canonicalization are mutually exclusive."
            )
        if enabled and self.equivariance_regularization_enabled():
            raise ValueError(
                "frame_averaging and equivariance_regularization are mutually exclusive."
            )
        return enabled

    def _frame_averaging_auxiliary_loss_mode(self) -> str:
        mode = self._cfg_get(
            self.frame_averaging, "auxiliary_loss_mode", "stochastic_per_graph"
        )
        if mode not in {"stochastic_per_graph", "average_over_frames"}:
            raise ValueError(
                "frame_averaging.auxiliary_loss_mode must be one of "
                "{'stochastic_per_graph', 'average_over_frames'}, "
                f"got {mode!r}."
            )
        return mode

    def _frame_averaging_weighting_mode(self) -> str:
        mode = str(self._cfg_get(self.frame_averaging, "weighting_mode", "uniform"))
        if mode not in {"uniform", "learned"}:
            raise ValueError(
                "frame_averaging.weighting_mode must be one of "
                "{'uniform', 'learned'}, "
                f"got {mode!r}."
            )
        return mode

    def _frame_averaging_schedule_progress(self) -> float:
        trainer = getattr(self, "trainer", None)
        max_epochs = (
            getattr(trainer, "max_epochs", None) if trainer is not None else None
        )
        return self.frame_averaging_settings.schedule_progress(
            current_epoch=int(self.current_epoch),
            max_epochs=max_epochs,
            fallback_max_epochs=int(self.niter + self.niter_decay),
        )

    def _frame_averaging_tau(self) -> float:
        trainer = getattr(self, "trainer", None)
        max_epochs = (
            getattr(trainer, "max_epochs", None) if trainer is not None else None
        )
        return self.frame_averaging_settings.tau(
            current_epoch=int(self.current_epoch),
            max_epochs=max_epochs,
            fallback_max_epochs=int(self.niter + self.niter_decay),
        )
    def _frame_averaging_energy_input_source(self) -> str:
        source = str(
            self._cfg_get(
                self.frame_averaging,
                "energy_input_source",
                "framed_batch",
            )
        )
        if source not in {"framed_batch", "embedding"}:
            raise ValueError(
                "frame_averaging.energy_input_source must be one of "
                "{'framed_batch', 'embedding'}, "
                f"got {source!r}."
            )
        return source

    def _frame_averaging_energy_training_cfg(
        self, key: str, default: Any = None
    ) -> Any:
        if bool(self._cfg_get(self.learned_canonicalization, "enabled", False)):
            cfg = self._cfg_get(self.learned_canonicalization, "energy_training", {})
            return self._cfg_get(cfg, key, default)
        cfg = self._cfg_get(self.frame_averaging, "energy_training", {})
        return self._cfg_get(cfg, key, default)


    def _energy_training_enabled(self) -> bool:
        return bool(
            self.energy_net is not None
            and self._frame_averaging_energy_training_cfg("enabled", True)
        )

    def _manual_energy_optimization_enabled(self) -> bool:
        return self._energy_training_enabled()

    def _energy_uniform_warmup_epochs(self) -> int:
        epochs = int(
            self._frame_averaging_energy_training_cfg("uniform_warmup_epochs", 0)
        )
        if epochs < 0:
            raise ValueError(
                "frame_averaging.energy_training.uniform_warmup_epochs must be non-negative, "
                f"got {epochs}."
            )
        return epochs

    def _energy_training_active_for_stage(self, stage: str | None) -> bool:
        if not self._energy_training_enabled():
            return False
        if self.learned_canonicalization_enabled():
            return True
        if stage in {"train", "val"} and int(self.current_epoch) < self._energy_uniform_warmup_epochs():
            return False
        return True

    def _energy_optimizer_lr_scale(self) -> float:
        scale = float(
            self._frame_averaging_energy_training_cfg("optimizer_lr_scale", 0.5)
        )
        if scale <= 0.0:
            raise ValueError(
                "frame_averaging.energy_training.optimizer_lr_scale must be positive, "
                f"got {scale}."
            )
        return scale

    def _energy_optimizer_weight_decay(self) -> float:
        return float(
            self._frame_averaging_energy_training_cfg("weight_decay", 1.0e-4)
        )


    def _frame_averaging_energy_scheduler_cfg(
        self, key: str, default: Any = None
    ) -> Any:
        if bool(self._cfg_get(self.learned_canonicalization, "enabled", False)):
            training_cfg = self._cfg_get(
                self.learned_canonicalization, "energy_training", {}
            )
            scheduler_cfg = self._cfg_get(training_cfg, "scheduler", {})
            return self._cfg_get(scheduler_cfg, key, default)
        return self.frame_averaging_settings.energy_scheduler_value(key, default)
    def _energy_grad_clip_val(self) -> float:
        return float(
            self._frame_averaging_energy_training_cfg("grad_clip_val", 1.0)
        )

    def configure_manual_gradient_clipping(
        self, trainer_gradient_clip_val: float | None
    ) -> None:
        if not self._manual_energy_optimization_enabled():
            self._manual_main_grad_clip_val = 0.0
            return

        clip_val = self._frame_averaging_energy_training_cfg(
            "main_grad_clip_val", trainer_gradient_clip_val
        )
        if clip_val is None:
            self._manual_main_grad_clip_val = 0.0
            return

        clip_val = float(clip_val)
        if clip_val < 0.0:
            raise ValueError(
                "frame_averaging.energy_training.main_grad_clip_val must be non-negative, "
                f"got {clip_val}."
            )
        self._manual_main_grad_clip_val = clip_val

    def _main_grad_clip_val(self) -> float:
        return float(self._manual_main_grad_clip_val)

    def _clip_optimizer_if_configured(self, optimizer, clip_val: float) -> None:
        if float(clip_val) <= 0.0:
            return
        self.clip_gradients(
            optimizer,
            gradient_clip_val=float(clip_val),
            gradient_clip_algorithm="norm",
        )

    def _energy_scheduler_warmup_epochs(self) -> int:
        epochs = int(self._frame_averaging_energy_scheduler_cfg("warmup_epochs", 5))
        if epochs <= 0:
            raise ValueError(
                "frame_averaging.energy_training.scheduler.warmup_epochs must be positive, "
                f"got {epochs}."
            )
        return epochs

    def _energy_scheduler_type(self) -> str:
        scheduler_type = str(
            self._frame_averaging_energy_scheduler_cfg("type", "cosine")
        )
        if scheduler_type not in {"cosine", "constant"}:
            raise ValueError(
                "frame_averaging.energy_training.scheduler.type must be one of "
                "{'cosine', 'constant'}, "
                f"got {scheduler_type!r}."
            )
        return scheduler_type

    def _energy_scheduler_min_lr_scale(self) -> float:
        scale = float(self._frame_averaging_energy_scheduler_cfg("min_lr_scale", 0.0))
        if scale < 0.0:
            raise ValueError(
                "frame_averaging.energy_training.scheduler.min_lr_scale must be non-negative, "
                f"got {scale}."
            )
        return scale

    def _frame_averaging_collapse_level(self) -> str:
        level = str(self._cfg_get(self.frame_averaging, "collapse_level", "output"))
        if level not in {"output", "encoder"}:
            raise ValueError(
                "frame_averaging.collapse_level must be one of "
                "{'output', 'encoder'}, "
                f"got {level!r}."
            )
        return level

    def _frame_averaging_uses_encoder_collapse(self) -> bool:
        return self._frame_averaging_collapse_level() == "encoder"

    def _frame_averaging_entropy_regularization_cfg(
        self, key: str, default: Any = None
    ) -> Any:
        cfg = self._cfg_get(self.frame_averaging, "entropy_regularization", {})
        return self._cfg_get(cfg, key, default)

    def frame_averaging_entropy_regularization_enabled(self) -> bool:
        return bool(
            self._frame_averaging_uses_learned_weights(stage="train")
            and self._frame_averaging_entropy_regularization_cfg("enabled", False)
        )

    def _frame_averaging_entropy_regularization_initial_coefficient(self) -> float:
        return float(
            self._frame_averaging_entropy_regularization_cfg(
                "initial_coefficient", 0.0
            )
        )

    def _frame_averaging_entropy_regularization_final_coefficient(self) -> float:
        return float(
            self._frame_averaging_entropy_regularization_cfg("final_coefficient", 0.0)
        )

    def _frame_averaging_entropy_regularization_progress(self) -> float:
        return self._frame_averaging_schedule_progress()

    def frame_averaging_entropy_regularization_coefficient(self) -> float:
        start = self._frame_averaging_entropy_regularization_initial_coefficient()
        end = self._frame_averaging_entropy_regularization_final_coefficient()
        progress = self._frame_averaging_entropy_regularization_progress()
        return start + (end - start) * progress

    def _frame_averaging_uses_learned_weights(self, stage: str | None = None) -> bool:
        if self._frame_averaging_weighting_mode() != "learned":
            return False
        if self.energy_net is None:
            return True
        return self._energy_training_active_for_stage(stage)

    def _frame_averaging_include_pca_frames(self) -> bool:
        return bool(self._cfg_get(self.frame_averaging, "include_pca_frames", True))

    def _frame_averaging_pca_frame_group(self) -> str:
        frame_group = str(
            self._cfg_get(self.frame_averaging, "pca_frame_group", "o3")
        )
        if frame_group not in {"o3", "so3"}:
            raise ValueError(
                "frame_averaging.pca_frame_group must be one of {'o3', 'so3'}, "
                f"got {frame_group!r}."
            )
        return frame_group

    def _frame_averaging_detach_energy_input(self) -> bool:
        gradient_routing = self._cfg_get(
            self.frame_averaging, "gradient_routing", {}
        )
        return bool(self._cfg_get(gradient_routing, "detach_energy_input", False))

    def _frame_averaging_include_identity_frame(self) -> bool:
        return bool(
            self._cfg_get(self.frame_averaging, "include_identity_frame", False)
        )

    def _frame_averaging_num_random_frames(self) -> int:
        count = int(self._cfg_get(self.frame_averaging, "num_random_frames", 0))
        if count < 0:
            raise ValueError(
                "frame_averaging.num_random_frames must be non-negative, "
                f"got {count}."
            )
        return count

    def _build_frame_averaging_batches(self, batch) -> FrameAveragingBatch:
        return build_frame_averaging_batch(
            batch,
            include_pca_frames=self._frame_averaging_include_pca_frames(),
            pca_frame_group=self._frame_averaging_pca_frame_group(),
            include_identity_frame=self._frame_averaging_include_identity_frame(),
            num_random_frames=self._frame_averaging_num_random_frames(),
            featuriser=getattr(self, "featuriser", None),
            propagate_dynamic_node_attrs=self._propagate_dynamic_node_attrs,
            reuse_rigid_transform_invariant_features=self._reuse_rigid_transform_invariant_features,
        )

    def _reuse_rigid_transform_invariant_features(
        self,
        *,
        source_batch,
        target_batch,
        samples: list[Any],
        repeats: int,
    ) -> bool:
        return reuse_rigid_transform_invariant_features(
            source_batch=source_batch,
            target_batch=target_batch,
            samples=samples,
            repeats=repeats,
            featuriser=getattr(self, "featuriser", None),
        )

    def _propagate_dynamic_node_attrs(self, batch, samples: list[Any]) -> None:
        propagate_dynamic_node_attrs(batch, samples)

    def _compute_frame_averaging_weights(
        self,
        *,
        framed_batch,
        samples: list[Any],
        frame_count: int,
        expanded_embedding: torch.Tensor | None = None,
        stage: str | None = None,
    ) -> torch.Tensor:
        num_graphs = len(samples)
        if not self._frame_averaging_uses_learned_weights(stage=stage):
            return uniform_frame_weights(
                num_graphs=num_graphs,
                frame_count=frame_count,
                device=framed_batch.pos.device,
                dtype=framed_batch.pos.dtype,
            )

        if self.energy_net is None:
            raise ValueError(
                "Learned frame averaging requires energy_encoder_config and "
                "energy_decoder_config."
            )

        embedding = None
        if self._frame_averaging_energy_input_source() == "embedding":
            if not torch.is_tensor(expanded_embedding):
                raise ValueError(
                    "frame_averaging.energy_input_source='embedding' requires "
                    "an expanded embedding tensor."
                )
            embedding = expanded_embedding
            if self._frame_averaging_detach_energy_input():
                embedding = embedding.detach()

        energy_batch = prepare_frame_energy_batch(
            framed_batch=framed_batch,
            embedding=embedding,
        )
        energy_output = self.energy_net(energy_batch)
        energy_level = energy_output.get(
            "level", getattr(self.energy_decoder, "output_level", None)
        )
        if energy_level != "graph":
            raise ValueError(
                "Learned frame averaging requires the energy decoder to return "
                f"graph-level outputs, got {energy_level!r}."
            )

        energies = energy_output["pred"]
        if energies.ndim == 2:
            if energies.shape[1] != 1:
                raise ValueError(
                    "Learned frame averaging expects scalar graph energies, got "
                    f"{tuple(energies.shape)}."
                )
            energies = energies.squeeze(-1)
        elif energies.ndim != 1:
            raise ValueError(
                "Learned frame averaging expects graph energies with shape "
                f"(B * K,) or (B * K, 1), got {tuple(energies.shape)}."
            )

        expected = num_graphs * frame_count
        if int(energies.shape[0]) != int(expected):
            raise ValueError(
                "Learned frame averaging energy output count mismatch: "
                f"expected {expected}, got {energies.shape[0]}."
            )

        weights = torch.softmax(
            -energies.reshape(num_graphs, frame_count) / self._frame_averaging_tau(),
            dim=1,
        )
        if not torch.isfinite(weights).all():
            raise ValueError("Non-finite frame weights produced by the energy module.")
        return weights

    def _frame_averaging_metadata(
        self,
        *,
        stage: str,
        frame_weights: torch.Tensor,
        frame_rotations: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        return frame_averaging_metadata(
            stage=stage,
            frame_weights=frame_weights,
            frame_rotations=frame_rotations,
        )

    def _collapse_frame_averaging_outputs(
        self,
        expanded_output: dict[str, Any],
        samples: list[Any],
        frame_count: int,
        frame_weights: torch.Tensor,
        stage: str,
        frame_rotations: torch.Tensor,
    ) -> dict[str, Any]:
        level = expanded_output.get("level", getattr(self.decoder, "output_level", None))
        if level is None:
            raise ValueError(
                "Frame averaging requires model outputs to include an output level."
            )

        output = {}
        output["pred"] = weighted_frame_tensor(
            expanded_output["pred"],
            samples,
            frame_count=frame_count,
            level=level,
            frame_weights=frame_weights,
        )
        output["level"] = level
        output.update(
            self._frame_averaging_metadata(
                stage=stage,
                frame_weights=frame_weights,
                frame_rotations=frame_rotations,
            )
        )

        aux_mode = self._frame_averaging_auxiliary_loss_mode()
        selected_frame_indices = None
        if aux_mode == "stochastic_per_graph":
            selected_frame_indices = sample_frame_indices(
                frame_weights=frame_weights,
                uses_learned_weights=self._frame_averaging_uses_learned_weights(stage=stage),
                frame_count=frame_count,
            )

        for key, value in expanded_output.items():
            if key in {"pred", "level"} or not torch.is_tensor(value):
                continue
            tensor_level = infer_frame_tensor_level(value, samples, frame_count)
            if tensor_level is None:
                continue
            if aux_mode == "average_over_frames":
                output[key] = average_frame_tensor(
                    value, samples, frame_count=frame_count, level=tensor_level
                )
            else:
                output[key] = select_frame_tensor(
                    value,
                    samples,
                    frame_count=frame_count,
                    level=tensor_level,
                    selected_frame_indices=selected_frame_indices,
                )
        return output

    def forward_for_stage(self, batch, stage: str):
        if self.direct_canonicalization_enabled():
            result = self._direct_canonicalization_result(batch)
            output = self.forward(result.batch)
            output.update(self._direct_canonicalization_metadata(result))
            return output, batch

        if self.learned_canonicalization_enabled():
            result = self._canonicalization_result(batch)
            output = self.forward(result.batch)
            output.update(self._canonicalization_metadata(result))
            return output, batch

        if not self.frame_averaging_enabled(stage):
            return self.forward(batch), batch

        frame_data = self._build_frame_averaging_batches(batch)
        if self._frame_averaging_uses_encoder_collapse():
            if not getattr(self.decoder, "supports_encoder_frame_averaging", False):
                raise ValueError(
                    "frame_averaging.collapse_level='encoder' requires a decoder "
                    "that declares supports_encoder_frame_averaging=True."
                )

            expanded_encoded = self.net.encode(frame_data.framed_batch)
            expanded_embedding = None
            if self._frame_averaging_energy_input_source() == "embedding":
                expanded_embedding = self.net.extract_embedding(expanded_encoded)
            frame_weights = self._compute_frame_averaging_weights(
                framed_batch=frame_data.framed_batch,
                samples=frame_data.samples,
                frame_count=frame_data.frame_count,
                expanded_embedding=expanded_embedding,
                stage=stage,
            )
            collapsed_encoded = collapse_non_equivariant_encoder_output(
                expanded_encoded=expanded_encoded,
                original_batch=batch,
                samples=frame_data.samples,
                frame_count=frame_data.frame_count,
                frame_weights=frame_weights,
                frame_rotations=frame_data.frame_rotations,
            )
            output = self.net.package_output(
                encoded=collapsed_encoded,
                decoded=self.net.decode(collapsed_encoded),
            )
            output.update(
                self._frame_averaging_metadata(
                    stage=stage,
                    frame_weights=frame_weights,
                    frame_rotations=frame_data.frame_rotations,
                )
            )
            return output, batch

        # Collapse at the output level
        expanded_output = self.forward(frame_data.framed_batch)
        expanded_embedding = None
        if self._frame_averaging_energy_input_source() == "embedding":
            expanded_embedding = expanded_output.get("embedding")
        frame_weights = self._compute_frame_averaging_weights(
            framed_batch=frame_data.framed_batch,
            samples=frame_data.samples,
            frame_count=frame_data.frame_count,
            expanded_embedding=expanded_embedding,
            stage=stage,
        )
        output = self._collapse_frame_averaging_outputs(
            expanded_output,
            frame_data.samples,
            frame_data.frame_count,
            frame_weights,
            stage,
            frame_data.frame_rotations,
        )
        return output, batch

    def frame_averaging_artifacts_for_sample(
        self, output: dict[str, Any], sample_idx: int
    ) -> dict[str, torch.Tensor]:
        artifacts = {}
        frame_weights = output.get("frame_weights")
        if torch.is_tensor(frame_weights) and sample_idx < frame_weights.shape[0]:
            artifacts["frame_weights"] = frame_weights[sample_idx].detach().cpu()

        frame_rotations = output.get("frame_rotations")
        if torch.is_tensor(frame_rotations) and sample_idx < frame_rotations.shape[0]:
            artifacts["frame_rotations"] = frame_rotations[sample_idx].detach().cpu()

        return artifacts

    def learned_canonicalization_artifacts_for_sample(
        self, output: dict[str, Any], sample_idx: int
    ) -> dict[str, torch.Tensor]:
        artifacts = {}
        for key in (
            "canonicalization_rotation",
            "canonicalization_seed_energies",
            "canonicalization_energy_history",
            "canonicalization_selected_seed",
            "canonicalization_eigengaps",
            "canonicalization_energy_gap",
        ):
            value = output.get(key)
            if torch.is_tensor(value) and sample_idx < value.shape[0]:
                artifacts[key] = value[sample_idx].detach().cpu()
        return artifacts

    def direct_canonicalization_artifacts_for_sample(
        self, output: dict[str, Any], sample_idx: int
    ) -> dict[str, torch.Tensor]:
        artifacts = {}
        for key in (
            "direct_canonicalization_rotation",
            "direct_canonicalization_raw_matrix",
            "direct_canonicalization_singular_values",
            "direct_canonicalization_projection_residual",
        ):
            value = output.get(key)
            if torch.is_tensor(value) and sample_idx < value.shape[0]:
                artifacts[key] = value[sample_idx].detach().cpu()
        return artifacts

    def frame_averaging_entropy_regularization(
        self, output: dict[str, Any]
    ) -> dict[str, torch.Tensor | float] | None:
        if not self.frame_averaging_entropy_regularization_enabled():
            return None

        diagnostics = self.frame_averaging_weight_diagnostics(output)
        if diagnostics is None:
            return None

        normalized_entropy = diagnostics["normalized_entropy"]
        entropy_deficit = 1.0 - normalized_entropy
        coefficient = self.frame_averaging_entropy_regularization_coefficient()
        return {
            "loss": entropy_deficit * coefficient,
            "entropy_deficit": entropy_deficit,
            "normalized_entropy": normalized_entropy,
            "coefficient": coefficient,
        }

    def frame_averaging_weight_diagnostics(
        self, output: dict[str, Any]
    ) -> dict[str, torch.Tensor] | None:
        frame_weights = output.get("frame_weights")
        if not torch.is_tensor(frame_weights):
            return None
        if frame_weights.ndim != 2:
            raise ValueError(
                "Frame-averaging weight diagnostics expects frame_weights "
                f"with shape (num_graphs, frame_count), got {tuple(frame_weights.shape)}."
            )
        if frame_weights.shape[1] <= 1:
            normalized_entropy = frame_weights.new_tensor(1.0)
            top_weight_margin = frame_weights.new_tensor(0.0)
        else:
            safe_weights = frame_weights.clamp_min(1.0e-12)
            entropy = -(safe_weights * safe_weights.log()).sum(dim=1)
            normalized_entropy = entropy / math.log(float(frame_weights.shape[1]))
            normalized_entropy = normalized_entropy.mean()
            top_two_weights = frame_weights.topk(k=2, dim=1).values
            top_weight_margin = (top_two_weights[:, 0] - top_two_weights[:, 1]).mean()

        diagnostics = {
            "normalized_entropy": normalized_entropy,
            "top_weight_margin": top_weight_margin,
        }
        if self._frame_averaging_include_identity_frame():
            pca_frame_count = 0
            if self._frame_averaging_include_pca_frames():
                pca_frame_count = 4 if self._frame_averaging_pca_frame_group() == "so3" else 8
            if pca_frame_count >= frame_weights.shape[1]:
                raise ValueError(
                    "Frame-averaging identity frame is missing from frame_weights: "
                    f"expected index {pca_frame_count}, got {frame_weights.shape[1]} frame(s)."
                )
            diagnostics["identity_weight"] = frame_weights[:, pca_frame_count].mean()
        return diagnostics

    def _transform_edges_to_pc(
        self, coords: torch.Tensor | np.ndarray, edges: torch.Tensor | np.ndarray
    ):
        """Extract unique vertices from edges matching a mask.

        Args:
            coords: (N, 3) array of vertex coordinates
            edges: (2, E) array of edge vertex indices

        Returns:
            torch.Tensor: Coordinates
        """
        if isinstance(coords, np.ndarray):
            coords = torch.tensor(coords, dtype=torch.float)
        if isinstance(edges, np.ndarray):
            edges = torch.tensor(edges, dtype=torch.long)

        if self.edge_to_pc == "midpoint":
            # Compute midpoints of edges
            midpoints = (coords[edges[0]] + coords[edges[1]]) / 2.0
            return midpoints
        elif self.edge_to_pc == "endpoints":
            # Return all endpoints
            vertices = torch.unique(edges.flatten())
            return coords[vertices]
        else:
            raise ValueError(f"Unknown edge_to_pc method: {self.edge_to_pc}")

    def evaluate(self, out, batch, stage: str = "val"):
        raise NotImplementedError("Subclasses should implement the evaluate method.")

    def _apply_post_process(
        self,
        *,
        batch,
        pred_labels: torch.Tensor,
        stage: str,
        label_kind: str,
        pred_logits: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.post_process is None or not self.post_process.should_run(stage):
            return pred_labels

        data_list = batch.to_data_list()
        counts = self._get_prediction_counts(data_list, label_kind=label_kind)
        if sum(counts) != int(pred_labels.shape[0]):
            raise ValueError(
                f"Prediction length {pred_labels.shape[0]} does not match per-mesh counts {counts}"
            )
        if pred_logits is not None and int(pred_logits.shape[0]) != int(
            pred_labels.shape[0]
        ):
            raise ValueError(
                "Prediction logits and labels must have matching leading dimensions"
            )

        refined_chunks = []
        start = 0
        for mesh, count in zip(data_list, counts):
            end = start + count
            mesh_logits = pred_logits[start:end] if pred_logits is not None else None
            request = PostProcessInput(
                mesh=mesh,
                labels=pred_labels[start:end],
                logits=mesh_logits,
                stage=stage,
                label_kind=label_kind,
                class_names=self.classes,
            )
            response = self.post_process(request)
            if response.labels.shape != pred_labels[start:end].shape:
                raise ValueError(
                    "Post-process returned labels with incompatible shape "
                    f"{tuple(response.labels.shape)} for expected shape {tuple(pred_labels[start:end].shape)}"
                )
            refined_chunks.append(
                response.labels.to(device=pred_labels.device, dtype=pred_labels.dtype)
            )
            start = end

        return torch.cat(refined_chunks, dim=0) if refined_chunks else pred_labels

    def _get_prediction_counts(self, data_list: list, label_kind: str) -> list[int]:
        counts = []
        for mesh in data_list:
            if label_kind == "vertex":
                counts.append(int(mesh.num_nodes))
            elif label_kind == "face":
                if not hasattr(mesh, "face"):
                    raise ValueError("Face predictions require face on each mesh")
                counts.append(int(mesh.face.shape[1]))
            elif label_kind == "edge":
                if not hasattr(mesh, "edge_index"):
                    raise ValueError("Edge predictions require edge_index on each mesh")
                counts.append(int(mesh.edge_index.shape[1]))
            else:
                raise ValueError(f"Unsupported label kind: {label_kind}")
        return counts

    def _aggregate_and_log_results(
        self, results: list, stage: str, batch_size: int = None
    ):
        """Aggregate results from evaluate and log metrics.

        Args:
            results: List of result dicts from evaluate()
            stage: One of 'train', 'val', 'test'
            batch_size: Number of samples in batch (optional, uses len(results) if not provided)

        Returns:
            Dictionary of aggregated metrics
        """
        # Infer batch size from results if not provided
        if batch_size is None:
            batch_size = len(results)

        # Collect all metric values across samples
        aggregated = {}

        # Extract metrics for each class
        for class_name in self.metric_class_names:
            for metric_name in self.metric_names:
                key = f"{stage}/{class_name}/{metric_name}"
                values = []
                for result in results:
                    if key in result and isinstance(result[key], list):
                        values.extend(result[key])
                    elif key in result:
                        values.append(result[key])

                if values:
                    avg_value = np.mean(values)
                    aggregated[key] = avg_value
                    self.log(
                        key,
                        avg_value,
                        on_step=False,
                        on_epoch=True,
                        prog_bar=False,
                        batch_size=batch_size,
                    )

        # Compute and log averages across classes
        for metric_name in self.metric_names:
            values = []
            for class_name in self.metric_class_names:
                key = f"{stage}/{class_name}/{metric_name}"
                if key in aggregated:
                    values.append(aggregated[key])

            if values:
                avg_value = np.mean(values)
                avg_key = f"{stage}/avg/{metric_name}"
                aggregated[avg_key] = avg_value
                self.log(
                    avg_key,
                    avg_value,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=True,
                    batch_size=batch_size,
                )

        return aggregated

    def on_validation_epoch_end(self):
        """Called at the end of validation epoch."""
        if not self.validation_step_outputs:
            return

        # Aggregate metrics dynamically
        log.info(f"\nValidation Results:")

        # Log per-class metrics
        for class_name in self.metric_class_names:
            for metric_name in self.metric_names:
                key = f"val/{class_name}/{metric_name}"
                values = [x.get(key) for x in self.validation_step_outputs if key in x]
                if values:
                    avg_value = np.mean(values)
                    log.info(
                        f"  {class_name.title()} {metric_name.title()}: {avg_value:.6f}"
                    )

        # Log cross-class averages
        for metric_name in self.metric_names:
            avg_key = f"val/avg/{metric_name}"
            values = [
                x.get(avg_key) for x in self.validation_step_outputs if avg_key in x
            ]
            if values:
                avg_value = np.mean(values)
                log.info(f"  Average {metric_name.title()}: {avg_value:.6f}")

        self.validation_step_outputs.clear()

    def on_test_epoch_end(self):
        """Called at the end of test epoch."""
        if not self.test_step_outputs:
            return

        # Aggregate metrics dynamically
        log.info(f"\nTest Results:")

        # Log per-class metrics
        for class_name in self.metric_class_names:
            for metric_name in self.metric_names:
                key = f"test/{class_name}/{metric_name}"
                values = [x.get(key) for x in self.test_step_outputs if key in x]
                if values:
                    avg_value = np.mean(values)
                    log.info(
                        f"  {class_name.title()} {metric_name.title()}: {avg_value:.6f}"
                    )

        # Log cross-class averages
        for metric_name in self.metric_names:
            avg_key = f"test/avg/{metric_name}"
            values = [x.get(avg_key) for x in self.test_step_outputs if avg_key in x]
            if values:
                avg_value = np.mean(values)
                log.info(f"  Average {metric_name.title()}: {avg_value:.6f}")

        self.test_step_outputs.clear()

    def _trainable_parameters_excluding_energy(self) -> list[torch.nn.Parameter]:
        energy_param_ids = set()
        if self.energy_net is not None:
            energy_param_ids = {id(param) for param in self.energy_net.parameters()}
        return [
            param
            for param in self.parameters()
            if param.requires_grad and id(param) not in energy_param_ids
        ]

    def _build_optimizer(self, params, *, lr: float, weight_decay: float):
        optim_name = self.optimizer_name.lower()
        optim_params = dict(
            params=params,
            lr=lr,
            betas=(self.beta1, 0.999),
            weight_decay=weight_decay,
        )
        if optim_name == "adamw":
            return torch.optim.AdamW(**optim_params)
        if optim_name == "adam":
            return torch.optim.Adam(**optim_params)
        raise ValueError(
            f"Unsupported optimizer_name='{self.optimizer_name}'. Expected one of ['adam', 'adamw']."
        )

    def _build_main_scheduler_config(self, optimizer):
        if self.lr_policy == "lambda":

            def lambda_rule(epoch):
                lr_l = 1.0 - max(0, epoch + 1 - self.niter) / float(
                    self.niter_decay + 1
                )
                return lr_l

            scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer, lr_lambda=lambda_rule
            )
            return {"scheduler": scheduler, "interval": "epoch", "frequency": 1}
        if self.lr_policy == "step":
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=self.lr_decay_iters,
                gamma=self.lr_step_gamma,
            )
            return {"scheduler": scheduler, "interval": "epoch", "frequency": 1}
        if self.lr_policy == "onecycle":
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=self.lr,
                total_steps=self.trainer.estimated_stepping_batches,
            )
            return {"scheduler": scheduler, "interval": "step"}
        if self.lr_policy == "plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=self.plateau_factor,
                threshold=self.plateau_threshold,
                patience=self.plateau_patience,
                min_lr=self.min_lr,
            )
            return {
                "scheduler": scheduler,
                "monitor": "train/loss_pred",
                "interval": "epoch",
                "frequency": 1,
            }
        return None

    def _build_energy_scheduler_config(self, optimizer):
        warmup_epochs = self._energy_scheduler_warmup_epochs()
        scheduler_type = self._energy_scheduler_type()
        min_lr_scale = self._energy_scheduler_min_lr_scale()

        def lambda_rule(epoch):
            if epoch < warmup_epochs:
                return float(epoch + 1)
            if scheduler_type == "constant":
                return float(warmup_epochs)
            max_epochs = getattr(getattr(self, "trainer", None), "max_epochs", None)
            if max_epochs is None or int(max_epochs) <= 0:
                max_epochs = int(self.niter + self.niter_decay)
            active_epochs = max(int(max_epochs) - self._energy_uniform_warmup_epochs(), 1)
            decay_epochs = max(active_epochs - warmup_epochs, 1)
            progress = min(max(float(epoch - warmup_epochs) / float(decay_epochs), 0.0), 1.0)
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            scaled = min_lr_scale + (1.0 - min_lr_scale) * cosine
            return float(warmup_epochs) * scaled

        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lambda_rule
        )
        return {"scheduler": scheduler, "interval": "epoch", "frequency": 1}

    def _energy_optimizer_active_this_step(self) -> bool:
        return self._energy_training_active_for_stage("train")

    def _manual_scheduler_list(self):
        schedulers = self.lr_schedulers()
        if schedulers is None:
            return []
        if isinstance(schedulers, (list, tuple)):
            return list(schedulers)
        return [schedulers]

    def _step_manual_epoch_schedulers(self) -> None:
        if not self._manual_energy_optimization_enabled():
            return
        schedulers = self._manual_scheduler_list()
        if not schedulers:
            return

        has_main_scheduler = bool(getattr(self, "_manual_main_scheduler_enabled", True))
        energy_scheduler_idx = 1 if has_main_scheduler else 0

        if has_main_scheduler:
            main_scheduler = schedulers[0]
            if self.lr_policy == "plateau":
                metric = self.trainer.callback_metrics.get("train/loss_pred")
                if metric is None:
                    metric = self.trainer.callback_metrics.get("train/loss")
                if metric is not None:
                    main_scheduler.step(metric)
            elif self.lr_policy != "onecycle":
                main_scheduler.step()

        if (
            len(schedulers) > energy_scheduler_idx
            and self._energy_optimizer_active_this_step()
        ):
            schedulers[energy_scheduler_idx].step()

    def on_train_epoch_end(self):
        self._step_manual_epoch_schedulers()

    def _maybe_manual_optimization_step(self, total_loss: torch.Tensor) -> torch.Tensor:
        if not self._manual_energy_optimization_enabled():
            return total_loss

        optimizers = self.optimizers()
        if not isinstance(optimizers, (list, tuple)):
            raise RuntimeError(
                "Manual energy optimization expected main and energy optimizers."
            )
        if len(optimizers) != 2:
            raise RuntimeError(
                f"Manual energy optimization expected 2 optimizers, got {len(optimizers)}."
            )
        main_optimizer, energy_optimizer = optimizers
        energy_active = self._energy_optimizer_active_this_step()

        main_optimizer.zero_grad()
        if energy_active:
            energy_optimizer.zero_grad()
        self.manual_backward(total_loss)

        self._clip_optimizer_if_configured(main_optimizer, self._main_grad_clip_val())
        main_optimizer.step()

        energy_grad_finite = all(
            parameter.grad is None or torch.isfinite(parameter.grad).all()
            for parameter in self.energy_net.parameters()
        )
        if energy_active and energy_grad_finite:
            self._clip_optimizer_if_configured(
                energy_optimizer,
                self._energy_grad_clip_val(),
            )
            energy_optimizer.step()
            self.log("train/energy_optimizer_active", 1.0, on_step=True, on_epoch=False, prog_bar=False)
            self.log("train/energy_optimizer_finite", 1.0, on_step=True, on_epoch=False, prog_bar=False)
        else:
            self.log("train/energy_optimizer_active", 0.0, on_step=True, on_epoch=False, prog_bar=False)
            self.log(
                "train/energy_optimizer_finite",
                float(energy_grad_finite),
                on_step=True,
                on_epoch=False,
                prog_bar=False,
            )

        if self.lr_policy == "onecycle" and bool(
            getattr(self, "_manual_main_scheduler_enabled", True)
        ):
            schedulers = self._manual_scheduler_list()
            if schedulers:
                schedulers[0].step()

        return total_loss.detach()

    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler."""
        main_optimizer = self._build_optimizer(
            self._trainable_parameters_excluding_energy(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        main_scheduler_config = self._build_main_scheduler_config(main_optimizer)
        self._manual_main_scheduler_enabled = main_scheduler_config is not None

        if not self._manual_energy_optimization_enabled():
            if main_scheduler_config is None:
                return main_optimizer
            return {
                "optimizer": main_optimizer,
                "lr_scheduler": main_scheduler_config,
            }

        energy_params = [
            param for param in self.energy_net.parameters() if param.requires_grad
        ]
        warmup_epochs = self._energy_scheduler_warmup_epochs()
        target_energy_lr = self.lr * self._energy_optimizer_lr_scale()
        energy_optimizer = torch.optim.AdamW(
            energy_params,
            lr=target_energy_lr / float(warmup_epochs),
            betas=(self.beta1, 0.999),
            weight_decay=self._energy_optimizer_weight_decay(),
        )
        energy_scheduler_config = self._build_energy_scheduler_config(energy_optimizer)

        optimizers = [main_optimizer, energy_optimizer]
        scheduler_configs = []
        if main_scheduler_config is not None:
            scheduler_configs.append(main_scheduler_config)
        scheduler_configs.append(energy_scheduler_config)
        return optimizers, scheduler_configs
