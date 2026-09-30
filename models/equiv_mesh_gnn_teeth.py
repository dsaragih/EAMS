"""PyTorch Lightning Module for Teeth mesh face-label prediction."""

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


class EquivMeshGNNTeethClassifier(BaseClassifierLightning):
    """LightningModule for Teeth meshes.

    Differences vs liver module:
    - Main supervision is on faces (batch.face_labels) rather than edges.
    - Optional embedding supervision remains on vertices.
    - No per-edge loss weights are loaded.
    - Background metrics are NOT ignored by default.
    - Test step writes face-level predictions into saved mesh objects.
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
        face_pool: str = "first",
        eval_use_batch_stats: bool = False,
        eval_diagnostics: bool = False,
        eval_diagnostics_topk: int = 5,
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
        self.face_pool = face_pool
        self.eval_use_batch_stats = eval_use_batch_stats
        self.eval_diagnostics = eval_diagnostics
        self.eval_diagnostics_topk = eval_diagnostics_topk

        # Accumulate per-mesh test metrics over an epoch and export to CSV.
        self._test_metrics_rows: list[dict[str, Any]] = []
        self._epoch_confusions: dict[str, Optional[torch.Tensor]] = {
            "val": None,
            "test": None,
        }

    def _collect_loss_inputs(
        self, batch: Batch, out_dict: dict[str, Any]
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        out_for_loss = {}
        for key, loss_obj in self.losses.items():
            output_key = getattr(loss_obj, "model_output_key", None) or key
            if output_key not in out_dict:
                raise KeyError(
                    "Model output is missing required loss key "
                    f"'{output_key}' for configured loss '{key}'."
                )
            if output_key == "pred":
                out_for_loss[key] = self._get_face_logits(batch, out_dict)
            else:
                out_for_loss[key] = out_dict[output_key]
        labels = {"pred": batch.face_labels, "embedding": batch.vertex_labels}
        return out_for_loss, labels

    def _compute_losses(
        self, batch: Batch, out_dict: dict[str, Any]
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        out_for_loss, labels = self._collect_loss_inputs(batch, out_dict)
        losses = self.compute_loss(out_for_loss, labels, data=batch)
        return losses, out_for_loss

    def _compute_eval_pred_loss(
        self,
        batch: Batch,
        pred_logits: torch.Tensor,
    ) -> torch.Tensor:
        loss_obj = self.losses["pred"]
        kwargs = {}
        if loss_obj.is_weighted_loss:
            kwargs["w_loss"] = None

        for attr in loss_obj.required_attributes:
            if hasattr(batch, attr):
                kwargs[attr] = getattr(batch, attr)
            else:
                raise ValueError(
                    f"Required attribute '{attr}' for loss 'pred' not found in batch."
                )

        if hasattr(batch, "batch") and "batch" not in kwargs:
            kwargs["batch"] = batch.batch

        supervision = self._resolve_loss_supervision(
            output_key="pred",
            loss_obj=loss_obj,
            supervision={"pred": batch.face_labels},
        )

        loss = loss_obj(pred_logits, supervision, **kwargs)
        if self.loss_coefficients and "pred" in self.loss_coefficients:
            loss = loss * self.loss_coefficients["pred"]
        return loss

    def _update_epoch_confusion(
        self, stage: str, pred_class: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        valid_mask = labels != -1
        valid_pred = pred_class[valid_mask].detach().cpu().long()
        valid_labels = labels[valid_mask].detach().cpu().long()
        num_classes = len(self.classes)

        confusion = torch.zeros((num_classes, num_classes), dtype=torch.long)
        if valid_labels.numel() > 0:
            flat_index = valid_labels * num_classes + valid_pred
            confusion.view(-1).index_add_(
                0,
                flat_index,
                torch.ones_like(flat_index, dtype=torch.long),
            )

        if self._epoch_confusions[stage] is None:
            self._epoch_confusions[stage] = confusion
        else:
            self._epoch_confusions[stage] += confusion
        return confusion

    def _log_prediction_distribution(
        self, stage: str, confusion: torch.Tensor, batch_weight: int
    ) -> None:
        total = int(confusion.sum().item())
        if total == 0:
            return

        gt_counts = confusion.sum(dim=1)
        pred_counts = confusion.sum(dim=0)
        active_pred_classes = int((pred_counts > 0).sum().item())
        dominant_pred_rate = float(pred_counts.max().item()) / float(total)

        self.log(
            f"{stage}/diag/active_pred_classes",
            active_pred_classes,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            batch_size=batch_weight,
        )
        self.log(
            f"{stage}/diag/dominant_pred_rate",
            dominant_pred_rate,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            batch_size=batch_weight,
        )

    def _log_epoch_confusion_summary(self, stage: str) -> None:
        confusion = self._epoch_confusions.get(stage)
        if confusion is None or int(confusion.sum().item()) == 0:
            return

        total = int(confusion.sum().item())
        gt_counts = confusion.sum(dim=1)
        pred_counts = confusion.sum(dim=0)

        logger.info(f"{stage.title()} prediction histogram:")
        for class_id, class_name in enumerate(self.classes):
            gt_rate = float(gt_counts[class_id].item()) / float(total)
            pred_rate = float(pred_counts[class_id].item()) / float(total)
            logger.info(
                "  {}: gt_rate={:.4f}, pred_rate={:.4f}",
                class_name,
                gt_rate,
                pred_rate,
            )

        off_diag = confusion.clone()
        off_diag.fill_diagonal_(0)
        flat = off_diag.view(-1)
        nonzero = torch.nonzero(flat, as_tuple=False).flatten()
        if nonzero.numel() == 0:
            logger.info(f"{stage.title()} top confusions: none")
            return

        values = flat[nonzero]
        topk = min(self.eval_diagnostics_topk, int(values.numel()))
        top_values, top_indices = torch.topk(values, k=topk)
        logger.info(f"{stage.title()} top confusions:")
        for value, index_in_nonzero in zip(top_values.tolist(), top_indices.tolist()):
            flat_index = int(nonzero[index_in_nonzero].item())
            gt_class = flat_index // len(self.classes)
            pred_class = flat_index % len(self.classes)
            row_total = max(int(gt_counts[gt_class].item()), 1)
            rate = float(value) / float(row_total)
            logger.info(
                "  gt={} predicted_as={} count={} rate_within_gt={:.4f}",
                self.classes[gt_class],
                self.classes[pred_class],
                value,
                rate,
            )

    def on_validation_epoch_start(self) -> None:
        self._epoch_confusions["val"] = None

    def _vertex_logits_to_face_logits(
        self,
        vertex_logits: torch.Tensor,
        face: torch.Tensor,
    ) -> torch.Tensor:
        if face.numel() == 0:
            return vertex_logits.new_empty((0, vertex_logits.shape[-1]))

        face_vertex_logits = vertex_logits[face]
        if self.face_pool == "first":
            return face_vertex_logits[0].contiguous()
        if self.face_pool == "mean":
            return face_vertex_logits.mean(dim=0)
        if self.face_pool == "max":
            return face_vertex_logits.max(dim=0).values
        raise ValueError(f"Unsupported face_pool mode: {self.face_pool}")

    def _get_face_logits(self, batch: Batch, out_dict: dict[str, Any]) -> torch.Tensor:
        pred = out_dict["pred"]
        level = out_dict.get("level")
        # breakpoint()
        if level == "face":
            return pred
        if level != "vertex":
            raise ValueError(f"Unsupported teeth prediction level: {level}")
        if not hasattr(batch, "face") or batch.face is None:
            raise ValueError("Face-supervised teeth training requires batch.face")
        return self._vertex_logits_to_face_logits(pred, batch.face.long())

    def _split_face_tensor(
        self, batch: Batch, values: torch.Tensor
    ) -> list[torch.Tensor]:
        data_list = batch.to_data_list()
        counts = [int(mesh.face.shape[1]) for mesh in data_list]
        if sum(counts) != int(values.shape[0]):
            raise ValueError(
                f"Face tensor length {values.shape[0]} does not match batched face counts {counts}"
            )

        chunks = []
        start = 0
        for count in counts:
            end = start + count
            chunks.append(values[start:end])
            start = end
        return chunks

    def _get_face_centroids(
        self, mesh_pos: torch.Tensor, mesh_face: torch.Tensor
    ) -> torch.Tensor:
        if mesh_face.numel() == 0:
            return mesh_pos.new_empty((0, mesh_pos.shape[-1]))
        return mesh_pos[mesh_face.long()].mean(dim=0)

    def on_test_epoch_start(self) -> None:
        self._test_metrics_rows.clear()
        self._epoch_confusions["test"] = None

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
        if self.eval_diagnostics:
            self._log_epoch_confusion_summary("test")
        super().on_test_epoch_end()

        # Write once in DDP.
        if hasattr(self, "trainer") and self.trainer is not None:
            if not self.trainer.is_global_zero:
                return

        if not self._test_metrics_rows:
            return

        os.makedirs(self.results_dir, exist_ok=True)
        csv_path = join(self.results_dir, "test_metrics.csv")

        # Union of all metric keys across samples.
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

    def training_step(self, batch: Batch, batch_idx: int):
        if self.timing_stats:
            self.timing_stats.start("training_step_total")

        if self.timing_stats:
            self.timing_stats.start("forward_pass")
        out_dict, loss_batch = self.forward_for_stage(batch, stage="train")
        if self.timing_stats:
            self.timing_stats.stop("forward_pass")

        # Use the encoder outputs as-is (e.g., pred, embedding) and configure
        # loss names accordingly via Hydra (loss_dict.{pred,embedding}).
        if self.timing_stats:
            self.timing_stats.start("compute_loss")
        losses, out_for_loss = self._compute_losses(loss_batch, out_dict)

        equivariance_transformed_loss = None
        equivariance_consistency_loss = None
        if self.equivariance_regularization_enabled():
            with self._timed_section("eq_build_batch"):
                transformed_batch = self.build_equivariance_batch(batch)
            with self._timed_section("eq_forward_transformed"):
                transformed_out_dict = self.forward(transformed_batch)
            with self._timed_section("eq_loss_transformed"):
                transformed_losses, transformed_out_for_loss = self._compute_losses(
                    transformed_batch, transformed_out_dict
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
        """Compute metrics per-mesh from face predictions."""

        if pred_class is None:
            pred_class = out.detach().max(1)[1]
        labels = batch.face_labels
        pred_chunks = self._split_face_tensor(batch, pred_class)
        label_chunks = self._split_face_tensor(batch, labels)
        data_list = batch.to_data_list()

        results = []
        for sample, mesh_pred, mesh_label in zip(data_list, pred_chunks, label_chunks):
            mesh_pos = self._get_face_centroids(sample.pos, sample.face)

            valid_mask = mesh_label != -1
            list_pred = mesh_pred[valid_mask]
            list_label = mesh_label[valid_mask]
            pos_valid = mesh_pos[valid_mask]

            log_dict = {}
            for class_id, class_name in self.iter_metric_classes():
                # Baseline-like behavior: only compute metrics for classes that
                # appear in the GT for this mesh.
                gt_present = (list_label == class_id).any()
                if not bool(gt_present.item()):
                    continue

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
        out_dict, loss_batch = self.forward_for_stage(batch, stage="val")
        out = self._get_face_logits(batch, out_dict)
        losses, _ = self._compute_losses(loss_batch, out_dict)
        if "pred" in losses:
            losses["pred"] = self._compute_eval_pred_loss(batch, out)
        total_loss = torch.stack([v for v in losses.values()]).sum()
        pred_class = self._apply_post_process(
            batch=batch,
            pred_labels=out.detach().max(1)[1],
            pred_logits=out.detach(),
            stage="val",
            label_kind="face",
        )
        if self.timing_stats:
            self.timing_stats.stop("val_forward_pass")

        if self.timing_stats:
            self.timing_stats.start("val_evaluate")
        results = self.evaluate(out, batch, stage="val", pred_class=pred_class)
        if self.timing_stats:
            self.timing_stats.stop("val_evaluate")

        batch_size = batch.num_graphs
        valid_faces = int((batch.face_labels != -1).sum().item())
        self.log(
            "val/loss",
            total_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch_size,
        )
        for key, value in losses.items():
            self.log(
                f"val/loss_{key}",
                value,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                batch_size=batch_size,
            )

        if self.eval_diagnostics:
            confusion = self._update_epoch_confusion(
                "val", pred_class, batch.face_labels
            )
            self._log_prediction_distribution("val", confusion, max(valid_faces, 1))

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
        out = self._get_face_logits(batch, out_dict)
        pred_logits = out.detach()
        pred_class = self._apply_post_process(
            batch=batch,
            pred_labels=pred_logits.argmax(dim=1),
            pred_logits=pred_logits,
            stage="test",
            label_kind="face",
        )
        results = self.evaluate(out, batch, stage="test", pred_class=pred_class)

        batch_size = batch.num_graphs
        valid_faces = int((batch.face_labels != -1).sum().item())
        if self.eval_diagnostics:
            confusion = self._update_epoch_confusion(
                "test", pred_class, batch.face_labels
            )
            self._log_prediction_distribution("test", confusion, max(valid_faces, 1))

        self._append_test_metric_rows(batch, results, batch_idx)

        if self.save_pt_results:
            # Save per-mesh data as a torch_geometric Data object.
            # This is more robust than saving raw label arrays when preprocessing
            # changes vertex ordering/count (e.g., pruning, pooling, subsampling).
            os.makedirs(self.results_dir, exist_ok=True)
            data_list = batch.to_data_list()
            face_logit_chunks = self._split_face_tensor(batch, pred_logits)
            face_pred_chunks = self._split_face_tensor(batch, pred_class)
            face_gt_chunks = self._split_face_tensor(batch, batch.face_labels)

            for local_idx in range(batch.num_graphs):
                sample_data = data_list[local_idx].cpu()

                # Attach predictions (and logits for optional downstream evaluation)
                sample_data.pred_face_logits = (
                    face_logit_chunks[local_idx].detach().cpu()
                )
                sample_data.pred_face_labels = (
                    face_pred_chunks[local_idx].detach().cpu()
                )

                # Ensure labels exist on the saved sample (on CPU)
                if (
                    hasattr(sample_data, "vertex_labels")
                    and sample_data.vertex_labels is not None
                ):
                    sample_data.vertex_labels = sample_data.vertex_labels.detach().cpu()
                if not hasattr(sample_data, "face_labels"):
                    sample_data.face_labels = face_gt_chunks[local_idx].detach().cpu()
                else:
                    sample_data.face_labels = sample_data.face_labels.detach().cpu()

                # Minimal metadata
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

        aggregated = self._aggregate_and_log_results(
            results, stage="test", batch_size=batch_size
        )
        self.test_step_outputs.append(aggregated)
        return aggregated

    def on_validation_epoch_end(self) -> None:
        if self.eval_diagnostics:
            self._log_epoch_confusion_summary("val")
        super().on_validation_epoch_end()
