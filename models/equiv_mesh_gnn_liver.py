"""PyTorch Lightning Module for MeshGraphCNN."""

import torch
from typing import Optional, Any
from omegaconf import DictConfig

from models.base_module import BaseClassifierLightning
import os
import csv
from os.path import join

import numpy as np

from torch_geometric.data import Batch
from torch_geometric.utils import scatter
from loguru import logger
from hydra.utils import instantiate


class EquivMeshGNNLiverClassifier(BaseClassifierLightning):
    """Lightning Module wrapping the EMNN model.

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
        # Learning rate scheduler
        lr_policy: str = "lambda",
        lr_decay_iters: int = 50,
        niter: int = 100,
        niter_decay: int = 500,
        # Loss
        batched_input: bool = False,
        features: dict = {},
        edge_to_pc: str = "midpoint",
        loss_dict: dict[str, dict] = {},
        loss_coefficients: dict[str, dict] = {},
        metrics_dict: list[dict] = None,
        # Classes
        classes: str = None,
        include_background_metrics: bool = False,
        # Paths
        train_lossweight_dir: str = None,
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
            classes=classes,
            include_background_metrics=include_background_metrics,
            train_lossweight_dir=train_lossweight_dir,
            results_dir=results_dir,
            **kwargs,
        )

        # Load edge weights for training (liver pipeline)
        self.w_dict = {}
        if train_lossweight_dir and os.path.exists(train_lossweight_dir):
            self._load_edge_weights(train_lossweight_dir)

        logger.info("Instantiating featuriser...")
        self.featuriser: torch.nn.Module = instantiate(features)
        logger.info(self.featuriser)

        # Accumulate per-mesh test metrics over an epoch and export to CSV.
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
        """Load per-edge loss weights from disk."""
        for weight_name in os.listdir(folder_path):
            if weight_name.endswith(".eseg"):
                w_name = weight_name.split("-save-weight.eseg")[0]
                w_path = os.path.join(folder_path, weight_name)
                weight = np.loadtxt(open(w_path, "r"), dtype="float64")
                self.w_dict[w_name] = torch.from_numpy(weight).float()

    def _prepare_edge_weights(self, batch: Batch, out_shape):
        """Prepare edge weight array from mesh filenames.

        Args:
            batch: Batch containing batch information and filenames
            out_shape: Tuple (total_num_edges, num_classes)

        Returns:
            w_array: Edge weights of shape [total_num_edges, num_classes]
        """
        w_array = torch.ones(out_shape, device=self.device, dtype=torch.float32)

        if not (hasattr(batch, "filename") and isinstance(batch.filename, list)):
            return w_array

        start = 0
        data_list = batch.to_data_list()
        for batch_idx, sample in enumerate(data_list):
            name = batch.filename[batch_idx].split(".obj")[0]
            num_edges_in_mesh = int(sample.edge_labels.shape[0])
            end = start + num_edges_in_mesh

            if name in self.w_dict and num_edges_in_mesh > 0:
                mesh_weights = self.w_dict[name]
                if mesh_weights.shape[0] >= num_edges_in_mesh:
                    mesh_weights = mesh_weights[:num_edges_in_mesh]
                elif hasattr(sample, "edge_label_inverse"):
                    unique_edge_count = int(sample.edge_label_inverse.max().item() + 1)
                    if mesh_weights.shape[0] < unique_edge_count:
                        raise ValueError(
                            "Edge weight count does not match liver edge labels: "
                            f"weights={mesh_weights.shape[0]}, directed_edges={num_edges_in_mesh}, unique_edges={unique_edge_count}."
                        )
                    mesh_weights = mesh_weights[:unique_edge_count][
                        sample.edge_label_inverse.long().cpu()
                    ]
                else:
                    raise ValueError(
                        "Directed liver edge weights require edge_label_inverse when stored weights are shorter than edge_labels."
                    )

                extended_w = mesh_weights.unsqueeze(1).repeat(1, out_shape[1])
                w_array[start:end] = extended_w.to(self.device)

            start = end

        return w_array

    def _set_liver_edge_label_views(self, batch: Batch) -> Batch:
        promoted_samples = []

        for sample in batch.to_data_list():
            directed_count = int(sample.edge_index.shape[1])
            undirected_edges = torch.sort(sample.edge_index, dim=0).values.t()
            _, inverse = torch.unique(undirected_edges, dim=0, return_inverse=True)
            unique_edge_count = (
                int(inverse.max().item() + 1) if inverse.numel() > 0 else 0
            )

            label_count = int(sample.edge_labels.shape[0])
            if label_count == unique_edge_count:
                unique_labels = sample.edge_labels.long()
                directed_labels = unique_labels[inverse]
            elif label_count == directed_count:
                directed_labels = sample.edge_labels.long()
                unique_labels_min = scatter(
                    directed_labels,
                    inverse,
                    dim=0,
                    dim_size=unique_edge_count,
                    reduce="min",
                )
                unique_labels_max = scatter(
                    directed_labels,
                    inverse,
                    dim=0,
                    dim_size=unique_edge_count,
                    reduce="max",
                )
                if not torch.equal(unique_labels_min, unique_labels_max):
                    raise ValueError(
                        "Directed liver edge labels disagree across edge directions."
                    )
                unique_labels = unique_labels_max
            else:
                raise ValueError(
                    "Liver edge labels must align with either directed or unique undirected edges: "
                    f"edge_labels={label_count}, directed_edges={directed_count}, unique_edges={unique_edge_count}."
                )

            sample.edge_label_inverse = inverse.long()
            sample.edge_labels = directed_labels.long()
            promoted_samples.append(sample)

        return Batch.from_data_list(promoted_samples)

    def _get_vertex2edge(self, batch: Batch) -> torch.Tensor:
        if hasattr(batch, "vertex2edge") and batch.vertex2edge is not None:
            return batch.vertex2edge.long()

        edge_indices = torch.arange(
            batch.edge_index.shape[1], device=batch.edge_index.device, dtype=torch.long
        )
        return torch.stack(
            [
                torch.cat([batch.edge_index[0], batch.edge_index[1]], dim=0),
                torch.cat([edge_indices, edge_indices], dim=0),
            ],
            dim=1,
        )

    def _derive_vertex_labels_from_edges(self, batch: Batch) -> torch.Tensor:
        vertex2edge = self._get_vertex2edge(batch)
        device = batch.edge_labels.device
        vertex_labels = torch.full(
            (batch.num_nodes,),
            -1,
            dtype=batch.edge_labels.dtype,
            device=device,
        )
        if vertex2edge.numel() == 0 or batch.edge_labels.numel() == 0:
            return vertex_labels

        incident_edge_labels = batch.edge_labels[vertex2edge[:, 1]]
        valid_mask = incident_edge_labels != -1
        if not valid_mask.any():
            return vertex_labels

        if self.classes is not None:
            num_classes = len(self.classes)
        else:
            num_classes = int(incident_edge_labels[valid_mask].max().item()) + 1

        vote_counts = torch.zeros(
            (batch.num_nodes, num_classes),
            dtype=torch.long,
            device=device,
        )
        vertex_indices = vertex2edge[valid_mask, 0]
        class_indices = incident_edge_labels[valid_mask].long()
        vote_counts.index_put_(
            (vertex_indices, class_indices),
            torch.ones_like(class_indices, dtype=vote_counts.dtype),
            accumulate=True,
        )

        max_votes, majority_vote = vote_counts.max(dim=1)
        tied = vote_counts.eq(max_votes.unsqueeze(1)).sum(dim=1) > 1
        resolved = max_votes > 0
        vertex_labels[resolved & ~tied] = majority_vote[resolved & ~tied]
        return vertex_labels

    def _collect_loss_inputs(
        self, batch: Batch, out_dict: dict[str, torch.Tensor]
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        out_for_loss = {}
        for key, loss_obj in self.losses.items():
            output_key = getattr(loss_obj, "model_output_key", None) or key
            if output_key not in out_dict:
                raise KeyError(
                    f"Model output is missing required loss key '{output_key}' for configured loss '{key}'."
                )
            out_for_loss[key] = out_dict[output_key]

        labels = {"pred": batch.edge_labels}
        if "embedding" in self.losses:
            if not hasattr(batch, "vertex_labels") or batch.vertex_labels is None:
                raise ValueError(
                    "Liver embedding loss requires derived `vertex_labels`, but none were found on the batch."
                )
            labels["embedding"] = batch.vertex_labels

        return out_for_loss, labels

    def _validate_loss_inputs(
        self,
        batch: Batch,
        out_for_loss: dict[str, torch.Tensor],
        labels: dict[str, torch.Tensor],
    ) -> None:
        pred = out_for_loss.get("pred")
        if pred is not None and pred.shape[0] != batch.edge_labels.shape[0]:
            raise ValueError(
                "Liver edge prediction shape mismatch: "
                f"pred has {pred.shape[0]} rows but edge_labels has {batch.edge_labels.shape[0]}."
            )

        embedding = out_for_loss.get("embedding")
        embedding_labels = labels.get("embedding")
        if embedding is not None and embedding_labels is not None:
            if embedding.shape[0] != embedding_labels.shape[0]:
                raise ValueError(
                    "Liver embedding supervision shape mismatch: "
                    f"embedding has {embedding.shape[0]} rows but vertex_labels has {embedding_labels.shape[0]}."
                )

        assignment_matrix = out_for_loss.get("assignment_matrix")
        if (
            assignment_matrix is not None
            and assignment_matrix.shape[0] != batch.num_nodes
        ):
            raise ValueError(
                "Regional assignment output must align with liver vertices: "
                f"assignment_matrix has {assignment_matrix.shape[0]} rows but batch has {batch.num_nodes} nodes."
            )

    def on_after_batch_transfer(self, batch: Batch, dataloader_idx: int = 0) -> Batch:
        batch = self._set_liver_edge_label_views(batch)
        # Apply featuriser to batch after transfer to device
        batch = self.featuriser(batch)
        if "embedding" in self.losses and (
            not hasattr(batch, "vertex_labels") or batch.vertex_labels is None
        ):
            batch.vertex_labels = self._derive_vertex_labels_from_edges(batch)
        return batch

    def forward(self, batch: Batch):
        """Forward pass through the network."""
        return self.net(batch)

    def training_step(self, batch: Batch, batch_idx):
        """Training step."""
        if self.timing_stats:
            self.timing_stats.start("training_step_total")

        # Forward pass
        if self.timing_stats:
            self.timing_stats.start("forward_pass")
        out_dict, loss_batch = self.forward_for_stage(batch, stage="train")
        if self.timing_stats:
            self.timing_stats.stop("forward_pass")

        out_for_loss, labels = self._collect_loss_inputs(loss_batch, out_dict)
        self._validate_loss_inputs(loss_batch, out_for_loss, labels)

        # Compute loss
        if self.timing_stats:
            self.timing_stats.start("compute_loss")

        w_array = self._prepare_edge_weights(loss_batch, out_for_loss["pred"].shape)
        losses = self.compute_loss(out_for_loss, labels, w_loss=w_array, data=loss_batch)

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
            self._validate_loss_inputs(
                transformed_batch, transformed_out_for_loss, transformed_labels
            )
            transformed_w_array = self._prepare_edge_weights(
                transformed_batch, transformed_out_for_loss["pred"].shape
            )
            with self._timed_section("eq_loss_transformed"):
                transformed_losses = self.compute_loss(
                    transformed_out_for_loss,
                    transformed_labels,
                    w_loss=transformed_w_array,
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

        # Get batch size (number of graphs in batch)
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
        """Evaluate model on a batch and compute metrics.

        Args:
            batch: Batch containing edge_label and other attributes
        Returns:
            Dictionary with per-sample metrics including Dice coefficients
        """
        # Get predictions (shape: edges_total,)
        if pred_class is None:
            pred_class = out.detach().max(1)[1]
        labels = batch.edge_labels
        data_list = batch.to_data_list()
        edge_counts = [int(sample.edge_index.shape[1]) for sample in data_list]

        if sum(edge_counts) != int(pred_class.shape[0]):
            raise ValueError(
                "Liver evaluation expects one prediction per directed edge: "
                f"sum(edge_counts)={sum(edge_counts)} but pred_class has {pred_class.shape[0]} rows."
            )
        if sum(edge_counts) != int(labels.shape[0]):
            raise ValueError(
                "Liver evaluation expects batched edge_labels to align with per-sample counts: "
                f"sum(edge_counts)={sum(edge_counts)} but labels has {labels.shape[0]} rows."
            )

        pred_chunks = torch.split(pred_class, edge_counts)
        label_chunks = torch.split(labels, edge_counts)

        # Store results for each sample
        results = []

        for sample, mesh_pred, mesh_label in zip(data_list, pred_chunks, label_chunks):
            mesh_edges = sample.edge_index
            if mesh_edges is None:
                raise ValueError("Liver evaluation requires edge_index on each sample")

            # Filter out invalid labels (-1)
            valid_mask = mesh_label != -1
            list_pred = mesh_pred[valid_mask]
            list_label = mesh_label[valid_mask]
            valid_mesh_edges = mesh_edges[:, valid_mask]

            log_dict = {}
            for class_id, class_name in self.iter_metric_classes():
                # Label-based metrics
                pred_binary = self._create_binary_mask(list_pred, class_id)
                label_binary = self._create_binary_mask(list_label, class_id)

                # Point cloud metrics
                pred_edges = list_pred == class_id
                gt_edges = list_label == class_id
                if pred_edges.any() and gt_edges.any():
                    # Get vertices from edges with class 1
                    pred_coords = self._transform_edges_to_pc(
                        sample.pos, valid_mesh_edges[:, pred_edges]
                    )
                    gt_coords = self._transform_edges_to_pc(
                        sample.pos, valid_mesh_edges[:, gt_edges]
                    )
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
                for class_name in self.metric_class_names:
                    key = f"{stage}/{class_name}/{metric_name}"
                    if key in log_dict:
                        metric_values.append(log_dict[key])
                if metric_values:
                    avg_value = sum(metric_values) / len(metric_values)
                    log_dict[f"{stage}/average/{metric_name}"] = avg_value

            log_dict.update(
                {
                    "list_pred": list_pred,
                    "list_label": list_label,
                }
            )

            results.append(log_dict)

        return results

    def validation_step(self, batch: Batch, batch_idx):
        """Validation step."""
        if self.timing_stats:
            self.timing_stats.start("val_step_total")

        # Forward pass
        if self.timing_stats:
            self.timing_stats.start("val_forward_pass")
        out_dict, _ = self.forward_for_stage(batch, stage="val")
        out = out_dict["pred"]
        pred_class = self._apply_post_process(
            batch=batch,
            pred_labels=out.detach().max(1)[1],
            pred_logits=out.detach(),
            stage="val",
            label_kind="edge",
        )
        if self.timing_stats:
            self.timing_stats.stop("val_forward_pass")

        # Evaluate the batch
        if self.timing_stats:
            self.timing_stats.start("val_evaluate")
        results = self.evaluate(out, batch, stage="val", pred_class=pred_class)
        if self.timing_stats:
            self.timing_stats.stop("val_evaluate")

        # Get batch size (number of graphs)
        batch_size = batch.num_graphs

        # Aggregate and log metrics
        aggregated = self._aggregate_and_log_results(
            results, stage="val", batch_size=batch_size
        )

        # Store for epoch end
        self.validation_step_outputs.append(aggregated)

        if self.timing_stats:
            self.timing_stats.stop("val_step_total")

            # Print timing every 10 steps
            self._maybe_report_timing("val", batch_idx, logger.info)

        return aggregated

    def test_step(self, batch: Batch, batch_idx):
        """Test step with result saving."""
        # Evaluate the batch
        out_dict, _ = self.forward_for_stage(batch, stage="test")
        out = out_dict["pred"]
        pred_logits = out.detach()
        pred_class = self._apply_post_process(
            batch=batch,
            pred_labels=pred_logits.argmax(dim=1),
            pred_logits=pred_logits,
            stage="test",
            label_kind="edge",
        )
        results = self.evaluate(out, batch, stage="test", pred_class=pred_class)

        self._append_test_metric_rows(batch, results, batch_idx)

        # Get batch size (number of graphs)
        batch_size = batch.num_graphs

        if self.save_pt_results:
            # Save per-mesh data as a torch_geometric Data object.
            # This is more robust than saving raw label arrays when preprocessing
            # changes edge ordering/count.
            os.makedirs(self.results_dir, exist_ok=True)
            data_list = batch.to_data_list()

            edge_counts = [int(sample.edge_labels.shape[0]) for sample in data_list]
            logit_chunks = torch.split(pred_logits, edge_counts)
            label_chunks = torch.split(pred_class, edge_counts)

            for local_idx, sample_data in enumerate(data_list):
                sample_data = sample_data.cpu()

                sample_data.pred_edge_logits = logit_chunks[local_idx].detach().cpu()
                sample_data.pred_edge_labels = label_chunks[local_idx].detach().cpu()

                if hasattr(sample_data, "edge_labels"):
                    sample_data.edge_labels = sample_data.edge_labels.detach().cpu()

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

                torch.save(sample_data, join(self.results_dir, f"{base_name}.pt"))

        # Aggregate and log metrics
        aggregated = self._aggregate_and_log_results(
            results, stage="test", batch_size=batch_size
        )

        # Store for epoch end
        self.test_step_outputs.append(aggregated)

        return aggregated
