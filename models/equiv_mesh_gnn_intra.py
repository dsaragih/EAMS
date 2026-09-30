"""PyTorch Lightning Module for Intra mesh vertex-label prediction."""

import os
import csv
from os.path import join
from typing import Any, Optional

import torch
from hydra.utils import instantiate
from loguru import logger
from omegaconf import DictConfig
from torch_geometric.data import Batch

from models.base_module import BaseClassifierLightning


class EquivMeshGNNIntraClassifier(BaseClassifierLightning):
    """LightningModule for Intra meshes.

    Neary identical to Teeth module
    """

    def __init__(
        self,
        # Network configuration (Hydra config dict or actual network instance)
        network: Optional[Any] = None,
        network_config: Optional[DictConfig] = None,
        # Optimizer
        lr: float = 0.01,
        beta1: float = 0.9,
        # Learning rate scheduler
        lr_policy: str = "lambda",
        lr_decay_iters: int = 50,
        niter: int = 100,
        niter_decay: int = 500,
        # Loss / features
        batched_input: bool = False,
        features: dict = {},
        edge_to_pc: str = "midpoint",
        loss_dict: dict[str, dict] = {},
        loss_coefficients: dict[str, dict] = {},
        metrics_dict: dict[str, dict] | None = None,
        # Classes
        classes: str = None,
        include_background_metrics: bool = True,
        # Paths
        train_lossweight_dir: Optional[str] = None,
        results_dir: str = "./results",
        **kwargs,
    ):

        super().__init__(
            network=network,
            network_config=network_config,
            lr=lr,
            beta1=beta1,
            lr_policy=lr_policy,
            lr_decay_iters=lr_decay_iters,
            niter=niter,
            niter_decay=niter_decay,
            batched_input=batched_input,
            edge_to_pc=edge_to_pc,
            loss_dict=loss_dict,
            loss_coefficients=loss_coefficients,
            metrics_dict=metrics_dict,
            train_lossweight_dir=train_lossweight_dir,
            results_dir=results_dir,
            classes=classes,
            include_background_metrics=include_background_metrics,
            **kwargs,
        )

        logger.info("Instantiating featuriser...")
        self.featuriser: torch.nn.Module = instantiate(features)
        logger.info(self.featuriser)

        # Accumulate per-mesh test metrics over an epoch and export to CSV.
        self._test_metrics_rows: list[dict[str, Any]] = []

    def on_test_epoch_start(self) -> None:
        self._test_metrics_rows.clear()

    def _append_test_metric_rows(
        self, batch: Batch, results: list[dict], batch_idx: int
    ) -> None:
        """Append one CSV row per mesh/sample in the current batch."""

        for local_idx, result in enumerate(results):
            base_name = f"batch{batch_idx}_sample{local_idx}"
            if hasattr(batch, "filename") and isinstance(batch.filename, list):
                if local_idx < len(batch.filename):
                    raw = os.path.basename(str(batch.filename[local_idx]))
                    base_name = os.path.splitext(raw)[0]
            elif hasattr(batch, "mesh_id") and isinstance(batch.mesh_id, list):
                if local_idx < len(batch.mesh_id):
                    base_name = os.path.splitext(
                        os.path.basename(str(batch.mesh_id[local_idx]))
                    )[0]

            row: dict[str, Any] = {"mesh_id": base_name}

            for k, v in result.items():
                if not isinstance(k, str) or not k.startswith("test/"):
                    continue
                if torch.is_tensor(v):
                    if v.numel() == 1:
                        row[k] = v.detach().cpu().item()
                    continue
                if isinstance(v, (int, float, str, bool)):
                    row[k] = v

            self._test_metrics_rows.append(row)

    def on_test_epoch_end(self) -> None:
        super().on_test_epoch_end()

        # Write once in DDP.
        if hasattr(self, "trainer") and self.trainer is not None:
            if not self.trainer.is_global_zero:
                return

        if not self._test_metrics_rows:
            return

        os.makedirs(self.results_dir, exist_ok=True)
        csv_path = join(self.results_dir, "test_metrics.csv")

        metric_keys = sorted(
            {k for row in self._test_metrics_rows for k in row.keys() if k != "mesh_id"}
        )
        fieldnames = ["mesh_id", *metric_keys]

        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in self._test_metrics_rows:
                writer.writerow(row)

    def on_after_batch_transfer(self, batch: Batch, dataloader_idx: int = 0) -> Batch:
        with self._timed_section("batch_transfer_featurise"):
            batch = self.featuriser(batch)
        return batch

    def forward(self, batch: Batch):
        return self.net(batch)

    def _collect_loss_inputs(
        self, batch: Batch, out_dict: dict[str, Any]
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        """Map configured loss names to the outputs required by each loss."""

        out_for_loss = {}
        for key, loss_obj in self.losses.items():
            output_key = getattr(loss_obj, "model_output_key", None) or key
            if output_key not in out_dict:
                raise KeyError(
                    "Model output is missing required loss key "
                    f"'{output_key}' for configured loss '{key}'."
                )
            out_for_loss[key] = out_dict[output_key]
        return out_for_loss, batch.vertex_labels

    def training_step(self, batch: Batch, batch_idx: int):
        if self.timing_stats:
            self.timing_stats.start("training_step_total")

        if self.timing_stats:
            self.timing_stats.start("forward_pass")
        out_dict, loss_batch = self.forward_for_stage(batch, stage="train")
        if self.timing_stats:
            self.timing_stats.stop("forward_pass")

        out_for_loss, labels = self._collect_loss_inputs(loss_batch, out_dict)

        if self.timing_stats:
            self.timing_stats.start("compute_loss")
        losses = self.compute_loss(out_for_loss, labels, data=loss_batch)

        equivariance_transformed_loss = None
        equivariance_consistency_loss = None
        if self.equivariance_regularization_enabled():
            with self._timed_section("eq_build_batch"):
                transformed_batch = self.build_equivariance_batch(batch)
            with self._timed_section("eq_forward_transformed"):
                transformed_out_dict = self.forward(transformed_batch)
            transformed_out_for_loss, transformed_labels = self._collect_loss_inputs(
                transformed_batch, transformed_out_dict
            )
            with self._timed_section("eq_loss_transformed"):
                transformed_losses = self.compute_loss(
                    transformed_out_for_loss,
                    transformed_labels,
                    data=transformed_batch,
                )
            equivariance_transformed_loss = torch.stack(
                [v for v in transformed_losses.values()]
            ).sum()
            equivariance_consistency_loss = self.compute_equivariance_consistency_loss(
                out_for_loss, transformed_out_for_loss
            )
        canonicalization_consistency = None
        if self.learned_canonicalization_enabled():
            with self._timed_section("canonicalization_consistency"):
                canonicalization_consistency = self.learned_canonicalization_consistency(
                    original_output=out_dict,
                    batch=batch,
                )
        direct_canonicalization_consistency = None
        if self.direct_canonicalization_consistency_enabled():
            with self._timed_section("direct_canonicalization_consistency"):
                direct_canonicalization_consistency = (
                    self.direct_canonicalization_consistency(
                        original_output=out_dict,
                        batch=batch,
                    )
                )
        direct_canonicalization_id_bias = None
        if self.direct_canonicalization_enabled():
            with self._timed_section("direct_canonicalization_id_bias"):
                direct_canonicalization_id_bias = self.direct_canonicalization_id_bias(
                    out_dict
                )
        if self.timing_stats:
            self.timing_stats.stop("compute_loss")

        batch_size = batch.num_graphs
        total_loss = torch.stack([v for v in losses.values()]).sum()
        if equivariance_transformed_loss is not None:
            total_loss = total_loss + (
                self.equivariance_transformed_supervision_weight()
                * equivariance_transformed_loss
            )
        if equivariance_consistency_loss is not None:
            total_loss = total_loss + (
                self.equivariance_consistency_weight() * equivariance_consistency_loss
            )
        if canonicalization_consistency is not None:
            total_loss = total_loss + (
                canonicalization_consistency["weight"]
                * canonicalization_consistency["loss"]
            )
        if direct_canonicalization_consistency is not None:
            total_loss = total_loss + (
                direct_canonicalization_consistency["weight"]
                * direct_canonicalization_consistency["loss"]
            )
        if self.direct_canonicalization_id_bias_enabled():
            total_loss = total_loss + (
                direct_canonicalization_id_bias["weight"]
                * direct_canonicalization_id_bias["loss"]
            )
        frame_averaging_entropy_reg = self.frame_averaging_entropy_regularization(
            out_dict
        )
        frame_averaging_weight_diagnostics = self.frame_averaging_weight_diagnostics(
            out_dict
        )
        if frame_averaging_entropy_reg is not None:
            total_loss = total_loss + frame_averaging_entropy_reg["loss"]
        self.log(
            "train/loss",
            total_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch_size,
        )
        for k, v in losses.items():
            self.log(
                f"train/loss_{k}",
                v,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                batch_size=batch_size,
            )
        if frame_averaging_entropy_reg is not None:
            self.log(
                "train/loss_frame_averaging_entropy",
                frame_averaging_entropy_reg["loss"],
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                batch_size=batch_size,
            )
            self.log(
                "train/frame_averaging_entropy_deficit",
                frame_averaging_entropy_reg["entropy_deficit"],
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                batch_size=batch_size,
            )
            self.log(
                "train/frame_averaging_entropy_coefficient",
                frame_averaging_entropy_reg["coefficient"],
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                batch_size=batch_size,
            )
        if frame_averaging_weight_diagnostics is not None:
            self.log(
                "train/frame_averaging_normalized_entropy",
                frame_averaging_weight_diagnostics["normalized_entropy"],
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                batch_size=batch_size,
            )
            self.log(
                "train/frame_averaging_top_weight_margin",
                frame_averaging_weight_diagnostics["top_weight_margin"],
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                batch_size=batch_size,
            )
            identity_weight = frame_averaging_weight_diagnostics.get("identity_weight")
            if identity_weight is not None:
                self.log(
                    "train/frame_averaging_identity_weight",
                    identity_weight,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=False,
                    batch_size=batch_size,
                )
        if equivariance_transformed_loss is not None:
            self.log(
                "train/loss_equivariance_transformed",
                equivariance_transformed_loss,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                batch_size=batch_size,
            )
        if equivariance_consistency_loss is not None:
            self.log(
                "train/loss_equivariance_consistency",
                equivariance_consistency_loss,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                batch_size=batch_size,
            )
        if canonicalization_consistency is not None:
            self.log(
                "train/loss_canonicalization_consistency",
                canonicalization_consistency["loss"],
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                batch_size=batch_size,
            )
            for key in ("position_rms", "normal_rms", "pose_error", "weight"):
                self.log(
                    f"train/canonicalization_{key}",
                    canonicalization_consistency[key],
                    on_step=False,
                    on_epoch=True,
                    prog_bar=False,
                    batch_size=batch_size,
                )
            energy_history = out_dict["canonicalization_energy_history"]
            self.log(
                "train/canonicalization_energy_initial",
                energy_history[:, 0].mean(),
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                batch_size=batch_size,
            )
            self.log(
                "train/canonicalization_energy_final",
                energy_history[:, -1].mean(),
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                batch_size=batch_size,
            )
            self.log(
                "train/canonicalization_energy_drop",
                (energy_history[:, 0] - energy_history[:, -1]).mean(),
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                batch_size=batch_size,
            )
            self.log(
                "train/canonicalization_energy_gap",
                out_dict["canonicalization_energy_gap"].mean(),
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                batch_size=batch_size,
            )
            for step in range(1, energy_history.shape[1]):
                self.log(
                    f"train/canonicalization_energy_drop_step_{step}",
                    (energy_history[:, step - 1] - energy_history[:, step]).mean(),
                    on_step=False,
                    on_epoch=True,
                    prog_bar=False,
                    batch_size=batch_size,
                )
            for axis in range(out_dict["canonicalization_eigengaps"].shape[1]):
                self.log(
                    f"train/canonicalization_eigengap_{axis}",
                    out_dict["canonicalization_eigengaps"][:, axis].mean(),
                    on_step=False,
                    on_epoch=True,
                    prog_bar=False,
                    batch_size=batch_size,
                )
            for seed in range(out_dict["canonicalization_seed_energies"].shape[1]):
                self.log(
                    f"train/canonicalization_selected_seed_{seed}",
                    (out_dict["canonicalization_selected_seed"] == seed).float().mean(),
                    on_step=False,
                    on_epoch=True,
                    prog_bar=False,
                    batch_size=batch_size,
                )
        if direct_canonicalization_consistency is not None:
            self.log(
                "train/loss_direct_canonicalization_consistency",
                direct_canonicalization_consistency["loss"],
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                batch_size=batch_size,
            )
            for key in ("position_rms", "normal_rms", "pose_error", "weight"):
                self.log(
                    f"train/direct_canonicalization_{key}",
                    direct_canonicalization_consistency[key],
                    on_step=False,
                    on_epoch=True,
                    prog_bar=False,
                    batch_size=batch_size,
                )
        if direct_canonicalization_id_bias is not None:
            self.log(
                "train/loss_direct_canonicalization_id_bias",
                direct_canonicalization_id_bias["loss"],
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                batch_size=batch_size,
            )
            self.log(
                "train/direct_canonicalization_id_bias_weight",
                direct_canonicalization_id_bias["weight"],
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                batch_size=batch_size,
            )

        if self.timing_stats:
            self.timing_stats.stop("training_step_total")
            self._maybe_report_timing("train", batch_idx, logger.info)

        return self._maybe_manual_optimization_step(total_loss)

    def evaluate(
        self,
        out: torch.Tensor,
        batch: Batch,
        stage: str = "val",
        pred_class: Optional[torch.Tensor] = None,
    ):
        """Compute metrics per-mesh from vertex predictions."""

        if pred_class is None:
            pred_class = out.detach().max(1)[1]
        labels = batch.vertex_labels

        node_batch = batch.batch
        batch_indices = torch.unique(node_batch)

        results = []
        for batch_idx in batch_indices:
            node_mask = node_batch == batch_idx

            mesh_pred = pred_class[node_mask]
            mesh_label = labels[node_mask]
            mesh_pos = batch.pos[node_mask]

            valid_mask = mesh_label != -1
            list_pred = mesh_pred[valid_mask]
            list_label = mesh_label[valid_mask]
            pos_valid = mesh_pos[valid_mask]

            log_dict = {}
            for class_id, class_name in self.iter_metric_classes():
                pred_binary = self._create_binary_mask(list_pred, class_id)
                label_binary = self._create_binary_mask(list_label, class_id)

                pred_pts = pos_valid[list_pred == class_id]
                gt_pts = pos_valid[list_label == class_id]
                if pred_pts.numel() == 0:
                    pred_pts = torch.empty((0, 3), device=self.device)
                if gt_pts.numel() == 0:
                    gt_pts = torch.empty((0, 3), device=self.device)

                self.update_metrics(
                    log_dict,
                    class_name,
                    stage,
                    pred_binary,
                    label_binary,
                    pred_pts,
                    gt_pts,
                )

            # Store label arrays for test-time saving
            log_dict.update({"list_pred": mesh_pred, "list_label": mesh_label})
            results.append(log_dict)

        return results

    def validation_step(self, batch: Batch, batch_idx: int):
        if self.timing_stats:
            self.timing_stats.start("val_step_total")

        if self.timing_stats:
            self.timing_stats.start("val_forward_pass")
        out_dict, _ = self.forward_for_stage(batch, stage="val")
        out = out_dict["pred"]
        pred_class = self._apply_post_process(
            batch=batch,
            pred_labels=out.detach().max(1)[1],
            pred_logits=out.detach(),
            stage="val",
            label_kind="vertex",
        )
        if self.timing_stats:
            self.timing_stats.stop("val_forward_pass")

        if self.timing_stats:
            self.timing_stats.start("val_evaluate")
        results = self.evaluate(out, batch, stage="val", pred_class=pred_class)
        if self.timing_stats:
            self.timing_stats.stop("val_evaluate")

        batch_size = batch.num_graphs
        aggregated = self._aggregate_and_log_results(
            results, stage="val", batch_size=batch_size
        )
        self.validation_step_outputs.append(aggregated)

        if self.timing_stats:
            self.timing_stats.stop("val_step_total")
            self._maybe_report_timing("val", batch_idx, logger.info)

        return aggregated

    def test_step(self, batch: Batch, batch_idx: int):
        out_dict, _ = self.forward_for_stage(batch, stage="test")
        out = out_dict["pred"]
        pred_logits = out.detach()
        pred_class = self._apply_post_process(
            batch=batch,
            pred_labels=pred_logits.argmax(dim=1),
            pred_logits=pred_logits,
            stage="test",
            label_kind="vertex",
        )
        results = self.evaluate(out, batch, stage="test", pred_class=pred_class)

        self._append_test_metric_rows(batch, results, batch_idx)

        if self.save_pt_results:
            # Save per-mesh data as a torch_geometric Data object.
            # This is more robust than saving raw label arrays when preprocessing
            # changes vertex ordering/count (e.g., pruning, pooling, subsampling).
            os.makedirs(self.results_dir, exist_ok=True)
            data_list = batch.to_data_list()

            node_batch = batch.batch

            for local_idx in range(batch.num_graphs):
                node_mask = node_batch == local_idx
                sample_data = data_list[local_idx].cpu()

                sample_data.pred_vertex_logits = pred_logits[node_mask].detach().cpu()
                sample_data.pred_vertex_labels = pred_class[node_mask].detach().cpu()

                if not hasattr(sample_data, "vertex_labels"):
                    sample_data.vertex_labels = (
                        batch.vertex_labels[node_mask].detach().cpu()
                    )
                else:
                    sample_data.vertex_labels = sample_data.vertex_labels.detach().cpu()

                sample_data.test_batch_idx = int(batch_idx)
                sample_data.test_sample_idx = int(local_idx)

                for key, value in self.frame_averaging_artifacts_for_sample(
                    out_dict, local_idx
                ).items():
                    setattr(sample_data, key, value)
                for key, value in self.learned_canonicalization_artifacts_for_sample(
                    out_dict, local_idx
                ).items():
                    setattr(sample_data, key, value)
                for key, value in self.direct_canonicalization_artifacts_for_sample(
                    out_dict, local_idx
                ).items():
                    setattr(sample_data, key, value)

                base_name = f"batch{batch_idx}_sample{local_idx}"
                if hasattr(batch, "filename") and isinstance(batch.filename, list):
                    if local_idx < len(batch.filename):
                        raw = os.path.basename(str(batch.filename[local_idx]))
                        base_name = os.path.splitext(raw)[0]
                elif hasattr(batch, "mesh_id") and isinstance(batch.mesh_id, list):
                    if local_idx < len(batch.mesh_id):
                        base_name = os.path.splitext(
                            os.path.basename(str(batch.mesh_id[local_idx]))
                        )[0]

                torch.save(sample_data, join(self.results_dir, f"{base_name}.pt"))

        batch_size = batch.num_graphs
        aggregated = self._aggregate_and_log_results(
            results, stage="test", batch_size=batch_size
        )
        self.test_step_outputs.append(aggregated)
        return aggregated
