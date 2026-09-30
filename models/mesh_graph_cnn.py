"""PyTorch Lightning Module for MeshGraphCNN."""

import csv
import os
from os.path import join
import torch
from typing import Optional, Any
from omegaconf import DictConfig

from models.base_module import BaseClassifierLightning
import numpy as np
import time
from loguru import logger


class MeshClassifierLightning(BaseClassifierLightning):
    """Lightning Module wrapping the MeshGraphCNN model.

    This module handles:
    - Model initialization
    - Forward pass
    - Training/validation/test step logic
    - Optimization configuration
    - Metrics logging
    """

    def __init__(
        self,
        # Network configuration (Hydra config dict or actual network instance)
        network: Optional[Any] = None,
        network_config: Optional[DictConfig] = None,
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
        classes: str = None,
        include_background_metrics: bool = False,
        # Paths
        train_lossweight_dir: str = None,
        results_dir: str = "./results",
        post_process: Optional[Any] = None,
        profile_training: bool = False,
        **kwargs,
    ):
        super().__init__(
            network=network,
            network_config=network_config,
            lr=lr,
            beta1=beta1,
            weight_decay=weight_decay,
            optimizer_name=optimizer_name,
            lr_policy=lr_policy,
            lr_decay_iters=lr_decay_iters,
            lr_step_gamma=lr_step_gamma,
            niter=niter,
            niter_decay=niter_decay,
            plateau_factor=plateau_factor,
            plateau_threshold=plateau_threshold,
            plateau_patience=plateau_patience,
            min_lr=min_lr,
            batched_input=batched_input,
            edge_to_pc=edge_to_pc,
            loss_dict=loss_dict,
            loss_coefficients=loss_coefficients,
            metrics_dict=metrics_dict,
            classes=classes,
            include_background_metrics=include_background_metrics,
            train_lossweight_dir=train_lossweight_dir,
            results_dir=results_dir,
            post_process=post_process,
            profile_training=profile_training,
            **kwargs,
        )

        self.w_dict = {}
        if train_lossweight_dir and os.path.exists(train_lossweight_dir):
            self._load_edge_weights(train_lossweight_dir)

        self._test_metrics_rows: list[dict[str, Any]] = []

    @staticmethod
    def _build_average_metric_row(
        rows: list[dict[str, Any]],
        *,
        id_key: str = "mesh_id",
        average_label: str = "average",
    ) -> dict[str, Any]:
        avg_row: dict[str, Any] = {id_key: average_label}
        metric_keys = sorted({k for row in rows for k in row.keys() if k != id_key})

        for key in metric_keys:
            values = [
                row[key] for row in rows if isinstance(row.get(key), (int, float))
            ]
            if values:
                avg_row[key] = float(np.mean(values))

        return avg_row

    def on_test_epoch_start(self) -> None:
        self._test_metrics_rows.clear()

    def _append_test_metric_rows(self, results: list[dict[str, Any]]) -> None:
        for local_idx, result in enumerate(results):
            mesh = result.get("mesh")
            base_name = f"sample{local_idx}"
            if mesh is not None and hasattr(mesh, "filename"):
                raw_name = os.path.basename(str(mesh.filename))
                base_name = os.path.splitext(raw_name)[0]

            row: dict[str, Any] = {"mesh_id": base_name}
            for key, value in result.items():
                if not isinstance(key, str) or not key.startswith("test/"):
                    continue
                if torch.is_tensor(value):
                    if value.numel() == 1:
                        row[key] = value.detach().cpu().item()
                    continue
                if isinstance(value, (int, float, str, bool)):
                    row[key] = value

            self._test_metrics_rows.append(row)

    def on_test_epoch_end(self) -> None:
        super().on_test_epoch_end()

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
        rows_to_write = [
            *self._test_metrics_rows,
            self._build_average_metric_row(self._test_metrics_rows),
        ]
        fieldnames = ["mesh_id", *metric_keys]

        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows_to_write:
                writer.writerow(row)

    def _load_edge_weights(self, folder_path: str):
        for weight_name in os.listdir(folder_path):
            if weight_name.endswith(".eseg"):
                w_name = weight_name.split("-save-weight.eseg")[0]
                w_path = os.path.join(folder_path, weight_name)
                weight = np.loadtxt(open(w_path, "r"), dtype="float64")
                self.w_dict[w_name] = torch.from_numpy(weight).float()

    def _prepare_edge_weights(self, meshes, out_shape):
        w_array = torch.ones(out_shape, device=self.device, dtype=torch.float32)
        if not self.w_dict:
            return w_array

        if out_shape[0] == 0:
            return w_array

        if len(out_shape) == 3:
            batch_size, num_classes, max_edges = out_shape
            for batch_idx, mesh in enumerate(meshes[:batch_size]):
                mesh_name = getattr(mesh, "filename", None)
                edge_count = min(int(getattr(mesh, "edges_count", 0)), max_edges)
                if mesh_name is None or edge_count == 0:
                    continue

                mesh_key = os.path.basename(str(mesh_name)).split(".obj")[0]
                if mesh_key not in self.w_dict:
                    continue

                mesh_weights = self.w_dict[mesh_key][:edge_count].to(self.device)
                w_array[batch_idx, :, :edge_count] = mesh_weights.unsqueeze(0).expand(
                    num_classes, -1
                )
            return w_array

        if len(out_shape) != 2:
            raise ValueError(
                f"Unsupported edge-logit shape for MeshGraphCNN weights: {tuple(out_shape)}"
            )

        offset = 0
        for mesh in meshes:
            mesh_name = getattr(mesh, "filename", None)
            edge_count = int(getattr(mesh, "edges_count", 0))
            if mesh_name is None or edge_count == 0:
                offset += edge_count
                continue
            mesh_key = os.path.basename(str(mesh_name)).split(".obj")[0]
            if mesh_key not in self.w_dict:
                offset += edge_count
                continue

            mesh_weights = self.w_dict[mesh_key][:edge_count].to(self.device)
            w_array[offset : offset + edge_count] = mesh_weights.unsqueeze(1).repeat(
                1, out_shape[1]
            )
            offset += edge_count

        return w_array

    def _collect_loss_inputs(
        self,
        *,
        pred: torch.Tensor,
        dilation: torch.Tensor,
        erosion: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        output_lookup = {
            "pred": pred,
            "all": pred,
            "dilation": dilation,
            "erosion": erosion,
        }
        out_for_loss = {}
        for key in self.losses.keys():
            if key not in output_lookup:
                raise KeyError(
                    f"Unsupported baseline loss key '{key}'. Expected one of {list(output_lookup.keys())}."
                )
            out_for_loss[key] = output_lookup[key]
        return out_for_loss

    def forward(self, edge_features, labels, meshes):
        """Forward pass through the network."""
        (
            out,
            low_out,
            low_lab,
            fe_low,
            rounded_tensor,
            coarse_tensor,
            d_out,
            e_out,
            avg_inference_time,
            a_w,
            b_w,
        ) = self.net(edge_features, labels, meshes)
        return (
            out,
            d_out,
            e_out,
            low_out,
            low_lab,
            rounded_tensor,
            coarse_tensor,
            a_w,
            b_w,
        )

    def training_step(self, batch, batch_idx):
        """Training step."""
        # Unpack batch
        edge_features = batch["edge_features"]
        labels = batch["label"]
        meshes = batch["mesh"]

        # Forward pass
        out, d_out, e_out, low_out, low_lab, rounded_tensor, coarse_tensor, a_w, b_w = (
            self.forward(edge_features, labels, meshes)
        )

        # Prepare edge weights
        w_array = self._prepare_edge_weights(meshes, out.shape)

        # Compute loss
        out_for_loss = self._collect_loss_inputs(
            pred=out,
            dilation=d_out,
            erosion=e_out,
        )
        losses = self.compute_loss(out_for_loss, labels, w_loss=w_array, data=None)

        # Log metrics
        primary_loss_key = "pred" if "pred" in losses else "all"
        self.log(
            "train/loss",
            losses[primary_loss_key],
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "train/loss_pred",
            losses[primary_loss_key],
            on_step=False,
            on_epoch=True,  
        )
        self.log(
            "train/loss_dilation",
            losses.get("dilation", torch.tensor(0.0)),
            on_step=False,
            on_epoch=True,
        )
        self.log(
            "train/loss_erosion",
            losses.get("erosion", torch.tensor(0.0)),
            on_step=False,
            on_epoch=True,
        )

        return losses[primary_loss_key]

    def evaluate(self, out, batch, stage: str = "val"):
        """Evaluate model on a batch and compute metrics.

        Args:
            batch: Dictionary containing edge_features, label, and mesh

        Returns:
            Dictionary with per-sample metrics including Dice and Chamfer distances
        """

        labels = batch["label"]
        meshes = batch["mesh"]
        # Get predictions
        pred_class = out.data.max(1)[1]

        # Store results for each sample
        results = []

        for file_i in range(labels.shape[0]):
            mesh = meshes[file_i]
            mask_re = labels[file_i] != -1

            # Extract masked predictions and labels
            list_pred = pred_class[file_i][mask_re].cpu().numpy()
            list_label = labels[file_i][mask_re].cpu().numpy()

            log_dict = {}
            for class_id, class_name in self.iter_metric_classes():
                # Label-based metrics
                pred_masked = pred_class[file_i][mask_re].clone()
                label_masked = labels[file_i][mask_re].clone()
                pred_binary = self._create_binary_mask(pred_masked, class_id)
                label_binary = self._create_binary_mask(label_masked, class_id)

                # Point cloud metrics
                pred_edges = (list_pred == class_id)[: len(mesh.uncorrupted_edges)]
                gt_edges = (list_label == class_id)[: len(mesh.uncorrupted_edges)]
                if pred_edges.any() and gt_edges.any():
                    pred_coords = self._transform_edges_to_pc(
                        mesh.vs, mesh.uncorrupted_edges[pred_edges].T
                    ).to(self.device)
                    gt_coords = self._transform_edges_to_pc(
                        mesh.vs, mesh.uncorrupted_edges[gt_edges].T
                    ).to(self.device)
                else:
                    pred_coords = torch.empty((0, 3), device=self.device)
                    gt_coords = torch.empty((0, 3), device=self.device)

                # breakpoint()

                self.update_metrics(
                    log_dict,
                    class_name,
                    stage,
                    pred_binary,
                    label_binary,
                    pred_coords,
                    gt_coords,
                )
            # Compute class averages
            for metric_name in self.metric_names:
                metric_values = []
                for class_name in self.classes[1:]:
                    key = f"{stage}/{class_name}/{metric_name}"
                    if key in log_dict:
                        metric_values.append(log_dict[key])
                if metric_values:
                    avg_value = sum(metric_values) / len(metric_values)
                    log_dict[f"{stage}/average/{metric_name}"] = avg_value

            # Log dict plus other info
            log_dict.update(
                {
                    "filename": mesh.filename,
                    "mesh": mesh,
                    "list_pred": list_pred,
                    "list_label": list_label,
                }
            )
            results.append(log_dict)

        return results

    def validation_step(self, batch, batch_idx):
        """Validation step."""
        # Unpack batch
        edge_features = batch["edge_features"]
        labels = batch["label"]
        meshes = batch["mesh"]

        # Forward pass
        out, d_out, e_out, _, _, rounded_tensor, coarse_tensor, a_w, b_w = self.forward(
            edge_features, labels, meshes
        )
        results = self.evaluate(out, batch, stage="val")

        # Aggregate and log metrics
        aggregated = self._aggregate_and_log_results(
            results, stage="val", batch_size=len(results)
        )

        # Store for epoch end
        self.validation_step_outputs.append(aggregated)

        # Return average metric for checkpointing
        return aggregated

    def test_step(self, batch, batch_idx):
        """Test step with result saving."""
        # Unpack batch
        edge_features = batch["edge_features"]
        labels = batch["label"]
        meshes = batch["mesh"]

        # Forward pass
        out, d_out, e_out, _, _, rounded_tensor, coarse_tensor, a_w, b_w = self.forward(
            edge_features, labels, meshes
        )
        results = self.evaluate(out, batch, stage="test")
        self._append_test_metric_rows(results)

        # Get additional predictions for saving
        pred_d_out = d_out.data.max(1)[1]
        pred_e_out = e_out.data.max(1)[1]
        dilation = rounded_tensor.data.max(1)[1]
        erosion = coarse_tensor.data.max(1)[1]

        # Save results per sample
        for i, result in enumerate(results):
            mesh = result.get("mesh")
            list_pred = result.get("list_pred")
            list_label = result.get("list_label")

            if mesh is not None and list_pred is not None and list_label is not None:
                # Get mask for this sample
                mask_re = batch["label"][i] != -1
                list_dilation = dilation[i][mask_re].cpu().numpy()
                list_erosion = erosion[i][mask_re].cpu().numpy()

                # Save results to files
                base_name = mesh.filename.split(".obj")[0]
                np.savetxt(
                    join(self.results_dir, f"{base_name}-pred.eseg"),
                    list_pred,
                    fmt="%d",
                )
                np.savetxt(
                    join(self.results_dir, f"{base_name}-label.eseg"),
                    list_label,
                    fmt="%d",
                )
                np.savetxt(
                    join(self.results_dir, f"{base_name}-dilation.eseg"),
                    list_dilation,
                    fmt="%d",
                )
                np.savetxt(
                    join(self.results_dir, f"{base_name}-erosion.eseg"),
                    list_erosion,
                    fmt="%d",
                )

        # Aggregate and log metrics
        aggregated = self._aggregate_and_log_results(
            results, stage="test", batch_size=len(results)
        )

        # Store for epoch end
        self.test_step_outputs.append(aggregated)

        return aggregated
